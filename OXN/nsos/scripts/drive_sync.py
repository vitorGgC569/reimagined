"""NSOS v11 — Local-side Drive sync daemon.

Polls Google Drive's /MyDrive/nsos_v11/runs/<run_name>/ folder and
mirrors new checkpoints + metrics into the local
OXN/nsos/scripts/live_distill_v11_colab/ directory.

This is the LOCAL half of the federated training loop:
    Colab    → trains, writes checkpoints to /content/drive/nsos_v11/runs/...
    Drive    → cloud storage (5TB on the user's Google AI Pro plan)
    LOCAL    → this script polls Drive, downloads new files, owns history

Authentication: uses Google Drive Desktop (the local mount under
G:\\My Drive\\ or C:\\Users\\<name>\\Google Drive\\) by default — zero
auth setup because the user's Drive is already mounted in Windows.

If Drive Desktop isn't installed, fall back to rclone via the
NSOS_RCLONE_REMOTE env var (e.g. "gdrive:nsos_v11").

Usage:
    # one-shot pull (no daemon)
    python drive_sync.py --once

    # daemon mode (polls every 5 min)
    python drive_sync.py --interval 300

    # specific run only
    python drive_sync.py --run v11_t4_run01_2026-05-16 --interval 60
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


# ── Drive Desktop default mount paths (Windows) ─────────────────────────
DRIVE_DESKTOP_CANDIDATES = [
    Path(r"G:\My Drive\nsos_v11"),
    Path(r"H:\My Drive\nsos_v11"),
    Path(os.path.expanduser(r"~\Google Drive\My Drive\nsos_v11")),
    Path(os.path.expanduser(r"~\My Drive\nsos_v11")),
]

LOCAL_ROOT = Path(__file__).resolve().parent / "live_distill_v11_colab"
STATE_FILE = LOCAL_ROOT / ".sync_state.json"


def find_drive_root() -> Optional[Path]:
    """Detect the Drive Desktop mount.  Override via NSOS_DRIVE_ROOT."""
    override = os.environ.get("NSOS_DRIVE_ROOT")
    if override:
        p = Path(override)
        if p.exists():
            print(f"[sync] using NSOS_DRIVE_ROOT={p}")
            return p
        print(f"[sync] WARNING: NSOS_DRIVE_ROOT={override} does not exist")
        return None

    for candidate in DRIVE_DESKTOP_CANDIDATES:
        if candidate.exists():
            print(f"[sync] detected Drive Desktop mount: {candidate}")
            return candidate

    return None


def find_rclone_remote() -> Optional[str]:
    """Check if rclone is available and a remote is configured."""
    if shutil.which("rclone") is None:
        return None
    remote = os.environ.get("NSOS_RCLONE_REMOTE")
    if not remote:
        return None
    print(f"[sync] using rclone remote: {remote}")
    return remote


# ── State tracking ──────────────────────────────────────────────────────
def load_state() -> dict:
    if not STATE_FILE.exists():
        return {"runs": {}}
    try:
        return json.loads(STATE_FILE.read_text("utf-8"))
    except json.JSONDecodeError:
        print(f"[sync] WARNING: {STATE_FILE} corrupt, resetting")
        return {"runs": {}}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_FILE)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Drive Desktop sync (no API calls — pure filesystem copy) ────────────
def sync_via_drive_desktop(drive_root: Path, run_filter: Optional[str],
                           state: dict) -> int:
    """Mirror new files from drive_root/runs/* to LOCAL_ROOT/<run>/.

    Returns count of newly downloaded files.
    """
    runs_dir = drive_root / "runs"
    if not runs_dir.exists():
        print(f"[sync] no runs/ in Drive yet ({runs_dir})")
        return 0

    new_count = 0
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_filter and run_dir.name != run_filter:
            continue

        local_run = LOCAL_ROOT / run_dir.name
        local_run.mkdir(parents=True, exist_ok=True)

        run_state = state["runs"].setdefault(run_dir.name, {})

        for src_file in run_dir.iterdir():
            if not src_file.is_file():
                continue
            stat = src_file.stat()
            key = src_file.name
            cached = run_state.get(key, {})
            if (cached.get("size") == stat.st_size and
                    cached.get("mtime") == stat.st_mtime):
                continue   # unchanged

            dst = local_run / src_file.name
            print(f"[sync] {run_dir.name}/{src_file.name}  "
                  f"({stat.st_size/1e6:.1f} MB)")
            shutil.copy2(src_file, dst)
            run_state[key] = {
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "downloaded_at": time.time(),
            }
            new_count += 1

    if new_count > 0:
        save_state(state)
    return new_count


# ── rclone fallback ─────────────────────────────────────────────────────
def sync_via_rclone(remote: str, run_filter: Optional[str]) -> int:
    """Use `rclone sync` (one-way Drive → local) for the runs/ tree.

    rclone handles its own change detection; we just count new files
    via the --stats-one-line-date output for telemetry."""
    src = f"{remote}/runs"
    if run_filter:
        src = f"{remote}/runs/{run_filter}"
        dst = LOCAL_ROOT / run_filter
    else:
        dst = LOCAL_ROOT
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[sync] rclone copy {src} -> {dst}")
    result = subprocess.run(
        ["rclone", "copy", src, str(dst),
         "--progress", "--transfers=4", "--checkers=8"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[sync] rclone failed: {result.stderr}", file=sys.stderr)
        return 0
    # Parse "Transferred: N" from stderr (rclone writes stats to stderr)
    new_count = 0
    for line in result.stderr.splitlines():
        if "Transferred:" in line and "files" in line:
            try:
                new_count = int(line.split()[1].split("/")[0])
                break
            except (IndexError, ValueError):
                pass
    return new_count


# ── Main loop ───────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true",
                        help="One-shot pull, no daemon loop.")
    parser.add_argument("--interval", type=int, default=300,
                        help="Daemon poll interval (seconds). Default 300 = 5min.")
    parser.add_argument("--run", default=None,
                        help="Sync only this run name (e.g. v11_t4_run01_2026-05-16).")
    args = parser.parse_args()

    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

    # Determine sync backend: Drive Desktop preferred, rclone fallback.
    drive_root = find_drive_root()
    rclone_remote = None if drive_root else find_rclone_remote()

    if not drive_root and not rclone_remote:
        print(
            "[sync] ERROR: neither Google Drive Desktop nor rclone configured.\n"
            "       Install Drive Desktop (https://www.google.com/drive/download/) OR\n"
            "       install rclone and set NSOS_RCLONE_REMOTE=gdrive:nsos_v11.",
            file=sys.stderr,
        )
        return 1

    state = load_state()

    def one_pass() -> int:
        if drive_root:
            return sync_via_drive_desktop(drive_root, args.run, state)
        else:
            return sync_via_rclone(rclone_remote, args.run)

    if args.once:
        n = one_pass()
        print(f"[sync] done — {n} new files")
        return 0

    print(f"[sync] daemon mode, interval={args.interval}s")
    print(f"[sync] local target: {LOCAL_ROOT}")
    print(f"[sync] Ctrl+C to stop")

    try:
        while True:
            t0 = time.time()
            n = one_pass()
            t_elapsed = time.time() - t0
            if n > 0:
                print(f"[sync] pulled {n} files in {t_elapsed:.1f}s; "
                      f"sleeping {args.interval}s")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[sync] stopped by user")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
