"""prepare_data.py — bake the data/ folder for the standalone bundle.

This script is run by you (the maintainer) on your dev machine, NOT by the
friend.  It downloads the source datasets, normalizes them into the layout
that trainer_main.py expects, copies the curriculum bundle, generates a
manifest with SHA256 hashes for integrity checking, and (optionally) zstd-
compresses large JSONL files to halve the shipping size.

Output structure:

    <output>/data/
    ├── bundle/
    │   └── tokenizer_8192.ox3
    ├── datasets/
    │   ├── source_a_001.jsonl.zst    (CulturaX shards, compressed)
    │   ├── source_a_002.jsonl.zst
    │   ├── source_b.jsonl            (BR-TaxQA-R, uncompressed for fast iter)
    │   └── source_c.jsonl            (BACEN FAQ)
    ├── config/runtime.json           (copied from this directory)
    └── manifest.json                 (sha256 of every file)

Note on naming: dataset filenames are intentionally generic so the bundle
gives away as little intent as possible.  Internally we know which is
which; from a casual inspector's view they're just opaque blobs.

Usage:
    python prepare_data.py --output ./build_standalone --culturax-gb 15
    python prepare_data.py --output D:/oxta_bundle --culturax-gb 30 --compress

Requires:
    pip install datasets huggingface_hub tqdm
    pip install zstandard          (optional, for --compress)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


HERE = Path(__file__).resolve().parent
REPO_NSOS_ROOT = HERE.parent.parent  # OXN/nsos


def sha256_of(path: Path, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(buf_size):
            h.update(chunk)
    return h.hexdigest()


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def run_download_step(culturax_gb: int, scratch_dir: Path) -> Path:
    """Invoke the existing download_datasets.py with reasonable settings."""
    download_script = REPO_NSOS_ROOT / "scripts" / "oxta_contabil" / "download_datasets.py"
    if not download_script.exists():
        sys.stderr.write(f"ERROR: download script not found at {download_script}\n")
        sys.exit(2)

    print(f"[step 1/4] downloading datasets to {scratch_dir} ...")
    cmd = [
        sys.executable, str(download_script),
        "--output-dir", str(scratch_dir),
        "--include", "culturax_ptbr,br_taxqa,bacen_faq,lener_br",
        "--target-bytes", f"{culturax_gb}GB",
        "--culturax-shards", str(max(20, culturax_gb * 4)),
    ]
    print("  $ " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode not in (0, 2):  # 2 = some dataset failed but others ok
        sys.stderr.write(f"ERROR: download_datasets exit={result.returncode}\n")
        sys.exit(3)

    return scratch_dir / "oxta_contabil" / "raw"


def normalize_culturax(raw_dir: Path, dest_dir: Path, compress: bool) -> List[Path]:
    """Copy CulturaX shards into dest with generic names + optional compression."""
    print(f"[step 2/4] normalizing CulturaX shards ...")
    shards = sorted(raw_dir.glob("culturax_ptbr_shard_*.jsonl"))
    if not shards:
        print(f"  no CulturaX shards found in {raw_dir} — skipping")
        return []

    out_files: List[Path] = []
    for idx, shard in enumerate(shards):
        dest_name = f"corpus_a_{idx:04d}.jsonl"
        dest_path = dest_dir / dest_name
        if compress:
            try:
                import zstandard as zstd
            except ImportError:
                print("  zstandard not available; falling back to uncompressed copy")
                shutil.copy2(shard, dest_path)
                out_files.append(dest_path)
                continue
            dest_path = dest_path.with_suffix(".jsonl.zst")
            cctx = zstd.ZstdCompressor(level=9)
            with shard.open("rb") as fin, dest_path.open("wb") as fout:
                cctx.copy_stream(fin, fout)
        else:
            shutil.copy2(shard, dest_path)
        out_files.append(dest_path)
        print(f"  shard {idx + 1}/{len(shards)}: "
              f"{shard.stat().st_size / 1024**2:.1f}MB → "
              f"{dest_path.stat().st_size / 1024**2:.1f}MB"
              + (" (compressed)" if compress else ""))
    return out_files


def normalize_br_taxqa(raw_dir: Path, dest_dir: Path) -> List[Path]:
    """Convert BR-TaxQA-R JSON files to plain .jsonl with {text} per line."""
    print(f"[step 2/4 cont] normalizing BR-TaxQA-R ...")
    src_dir = raw_dir / "br_taxqa"
    if not src_dir.exists():
        print(f"  not found at {src_dir} — skipping")
        return []

    out_path = dest_dir / "corpus_b.jsonl"
    count = 0
    with out_path.open("w", encoding="utf-8") as fout:
        # Questions Q&A
        q_files = sorted(src_dir.glob("questions_QA_*.json"))
        if q_files:
            data = json.loads(q_files[-1].read_text(encoding="utf-8"))
            items = data["questions"] if isinstance(data, dict) and "questions" in data \
                    else data if isinstance(data, list) \
                    else list(data.values())
            for item in items:
                if not isinstance(item, dict):
                    continue
                q = (item.get("question_text") or item.get("question") or "").strip()
                a = item.get("answer") or item.get("answer_cleaned") or ""
                if isinstance(a, list):
                    a = "\n".join(str(x) for x in a)
                text = (q + "\n\n" + str(a).strip()).strip()
                if len(text) >= 200:
                    fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    count += 1
        # CARF rulings
        for carf_file in sorted(src_dir.glob("acordaos_CARF_*.json")):
            data = json.loads(carf_file.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else list(data.values()) \
                    if isinstance(data, dict) else []
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = (item.get("text") or item.get("filedata") or "").strip()
                if len(text) >= 200:
                    fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    count += 1
        # Referred legal docs
        for ref_file in sorted(src_dir.glob("referred_legal_documents_*.json")):
            data = json.loads(ref_file.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else list(data.values()) \
                    if isinstance(data, dict) else []
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = (item.get("filedata") or item.get("text") or "").strip()
                if len(text) >= 200:
                    fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    count += 1

    print(f"  wrote {count} documents → {out_path.relative_to(dest_dir.parent.parent)}")
    return [out_path]


def normalize_simple_datasets(raw_dir: Path, dest_dir: Path) -> List[Path]:
    """BACEN FAQ + LeNER-Br: try load_from_disk, write {text} JSONL."""
    print(f"[step 2/4 cont] normalizing BACEN + LeNER ...")
    out_files: List[Path] = []
    try:
        from datasets import load_from_disk
    except ImportError:
        print("  datasets library missing — skipping BACEN + LeNER")
        return out_files

    for name, generic_name in [("bacen_faq", "corpus_c.jsonl"),
                                 ("lener_br", "corpus_d.jsonl")]:
        src = raw_dir / name
        if not src.exists():
            continue
        try:
            ds = load_from_disk(str(src))
        except Exception as exc:
            print(f"  {name} load failed: {exc}")
            continue
        out_path = dest_dir / generic_name
        count = 0
        with out_path.open("w", encoding="utf-8") as fout:
            for row in ds:
                parts = []
                for key in ("question", "pergunta", "q"):
                    if key in row and row[key]:
                        parts.append(str(row[key]).strip()); break
                for key in ("answer", "resposta", "a", "body", "text"):
                    if key in row and row[key]:
                        parts.append(str(row[key]).strip()); break
                if not parts:
                    tokens = row.get("tokens") or row.get("words") or []
                    if tokens:
                        parts = [" ".join(str(t) for t in tokens)]
                text = "\n\n".join(p for p in parts if p)
                if len(text) >= 200:
                    fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    count += 1
        print(f"  {name} → {generic_name}: {count} docs")
        out_files.append(out_path)
    return out_files


def copy_bundle(dest_root: Path) -> Path:
    """Copy the distillation_bundle_v11 tokenizer file into bundle/."""
    print(f"[step 3/4] copying tokenizer bundle ...")
    src_bundle = REPO_NSOS_ROOT / "scripts" / "distillation_bundle_v11"
    src_tokenizer = src_bundle / "tokenizer_8192.ox3"
    if not src_tokenizer.exists():
        print(f"  WARNING: bundle not found at {src_bundle}")
        print(f"  You need to extract distillation_bundle_v11.zip into {src_bundle.parent}/")
        print(f"  Continuing without bundle — trainer will fail at runtime until you fix this.")
        return src_bundle
    dest_bundle = dest_root / "bundle"
    dest_bundle.mkdir(parents=True, exist_ok=True)
    dest_tok = dest_bundle / "tokenizer_8192.ox3"
    shutil.copy2(src_tokenizer, dest_tok)
    print(f"  tokenizer: {src_tokenizer.stat().st_size / 1024**2:.1f}MB → {dest_tok.relative_to(dest_root.parent)}")
    return dest_tok


def copy_runtime_config(dest_root: Path) -> Path:
    print(f"[step 3/4 cont] copying runtime.json ...")
    src = HERE / "runtime.json"
    dest = dest_root / "config" / "runtime.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def write_manifest(data_root: Path) -> Path:
    print(f"[step 4/4] generating manifest with SHA256 hashes ...")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "1.0",
        "files": [],
    }
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        rel = path.relative_to(data_root).as_posix()
        size = path.stat().st_size
        h = sha256_of(path)
        manifest["files"].append({
            "path": rel,
            "size": size,
            "sha256": h,
        })
        print(f"  + {rel} ({human_size(size)})")

    manifest_path = data_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  manifest written to {manifest_path}")
    return manifest_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True,
                    help="Build directory (data/ subfolder will be created here)")
    ap.add_argument("--culturax-gb", type=int, default=15,
                    help="CulturaX sample size in GB (default 15; use 30 for full bundle)")
    ap.add_argument("--compress", action="store_true",
                    help="Compress CulturaX shards with zstd (saves ~40%% size)")
    ap.add_argument("--skip-download", action="store_true",
                    help="Skip download step (use cached raw datasets if already present)")
    args = ap.parse_args()

    output_root = args.output.resolve()
    data_root = output_root / "data"
    datasets_root = data_root / "datasets"
    datasets_root.mkdir(parents=True, exist_ok=True)

    print(f"=== prepare_data.py ===")
    print(f"output:        {output_root}")
    print(f"culturax_gb:   {args.culturax_gb}")
    print(f"compress:      {args.compress}")
    print()

    # 1) download
    scratch_dir = output_root / "_scratch"
    scratch_dir.mkdir(exist_ok=True)
    if args.skip_download:
        raw_dir = scratch_dir / "oxta_contabil" / "raw"
        print(f"[step 1/4] skipping download (using existing {raw_dir})")
    else:
        raw_dir = run_download_step(args.culturax_gb, scratch_dir)

    # 2) normalize each dataset family
    normalize_culturax(raw_dir, datasets_root, args.compress)
    normalize_br_taxqa(raw_dir, datasets_root)
    normalize_simple_datasets(raw_dir, datasets_root)

    # 3) bundle + config
    copy_bundle(data_root)
    copy_runtime_config(data_root)

    # 4) manifest
    write_manifest(data_root)

    # Summary
    total = sum(f.stat().st_size for f in data_root.rglob("*") if f.is_file())
    print()
    print(f"=== DONE ===")
    print(f"  data/ total size: {human_size(total)}")
    print(f"  next step:        run build_standalone.bat / .sh")

    return 0


if __name__ == "__main__":
    sys.exit(main())
