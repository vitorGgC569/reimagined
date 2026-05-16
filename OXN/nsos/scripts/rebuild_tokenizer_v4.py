"""
rebuild_tokenizer_v4.py
------------------------
Creates distillation_bundle_v4 by keeping all data files from v3
(including the Wikipedia phase3) and rebuilding ONLY the tokenizer.

Root cause of v8's C++ token generation:
  - v3 tokenizer was built on 2026-05-12 from old phase3 (NSOS C++ code)
  - phase3 was replaced with Wikipedia on 2026-05-14
  - Tokenizer was NEVER rebuilt → C++ tokens remained in vocabulary
  - v8 trained with a C++-biased tokenizer → generates code tokens at inference

Fix:
  - Copy all data files from v3 (Wikipedia phase3 included)
  - Rebuild tokenizer from scratch on the current data
  - The new BPE will learn English patterns instead of C++ identifiers
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPTS   = Path(__file__).resolve().parent
V3_DIR    = SCRIPTS / "distillation_bundle_v3"
V4_DIR    = SCRIPTS / "distillation_bundle_v4"
TARGET_VOCAB = 8192

# ── Add scripts dir to path for curriculum lib ────────────────────────────
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from nsos_curriculum_lib import (
    build_tokenizer_bundle,
    read_jsonl,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 60)
    print("rebuild_tokenizer_v4.py")
    print("=" * 60)
    print(f"Source:      {V3_DIR}")
    print(f"Destination: {V4_DIR}")
    print(f"Target vocab: {TARGET_VOCAB}")
    print()

    # 1. Create v4 directory
    V4_DIR.mkdir(parents=True, exist_ok=True)
    (V4_DIR / "data").mkdir(exist_ok=True)

    # 2. Copy all data files from v3
    v3_data = V3_DIR / "data"
    v4_data = V4_DIR / "data"
    print("Copying data files...")
    copied_files = []
    for src in sorted(v3_data.glob("*.jsonl")):
        dst = v4_data / src.name
        shutil.copy2(src, dst)
        rows = len(src.read_text("utf-8").splitlines())
        sha = sha256_file(dst)
        copied_files.append({
            "src": src.name,
            "rows": rows,
            "sha256": sha,
        })
        print(f"  ✓ {src.name} ({rows} rows)")

    # 3. Build new manifest with correct SHA256 for all files
    print("\nUpdating curriculum manifest...")
    old_manifest = json.loads((V3_DIR / "curriculum_manifest.json").read_text("utf-8"))

    # Rebuild phase entries with correct SHA256 (v3 SHA might be stale for phase3)
    phases = []
    for phase in old_manifest["phases"]:
        train_file = phase["train_file"].replace("data\\", "data\\")
        eval_file = phase["eval_file"].replace("data\\", "data\\")
        train_path = v4_data / Path(train_file.replace("data\\", "")).name
        eval_path = v4_data / Path(eval_file.replace("data\\", "")).name

        train_sha = sha256_file(train_path) if train_path.exists() else ""
        eval_sha = sha256_file(eval_path) if eval_path.exists() else ""
        train_rows = len(train_path.read_text("utf-8").splitlines()) if train_path.exists() else 0
        eval_rows = len(eval_path.read_text("utf-8").splitlines()) if eval_path.exists() else 0

        phases.append({
            "name": phase["name"],
            "train_file": f"data\\{train_path.name}",
            "eval_file": f"data\\{eval_path.name}",
            "train_samples": train_rows,
            "eval_samples": eval_rows,
            "train_sha256": train_sha,
            "eval_sha256": eval_sha,
        })
        print(f"  {phase['name']}: train={train_rows}, eval={eval_rows}")

    new_manifest = {
        "seed": old_manifest["seed"],
        "special_tokens": old_manifest["special_tokens"],
        "phases": phases,
        "_note": "v4: rebuilt tokenizer from Wikipedia-phase3 data. v3 tokenizer was built before Wikipedia replaced C++ code in phase3.",
    }
    (V4_DIR / "curriculum_manifest.json").write_text(
        json.dumps(new_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("  ✓ curriculum_manifest.json written")

    # 4. Rebuild the tokenizer from the current data (Wikipedia included)
    print(f"\nRebuilding tokenizer (vocab={TARGET_VOCAB})...")
    print("  [This reads all phases including Wikipedia phase3]")
    tok_path = build_tokenizer_bundle(V4_DIR, target_vocab=TARGET_VOCAB)
    print(f"  ✓ Tokenizer written: {tok_path}")

    # 5. Check what the new tokenizer looks like
    tokenizer_json = V4_DIR / f"tokenizer_{TARGET_VOCAB}.json"
    if tokenizer_json.exists():
        meta = json.loads(tokenizer_json.read_text("utf-8"))
        print(f"\nTokenizer stats:")
        print(f"  Learned merges:        {meta.get('learned_merges', 'n/a')}")
        print(f"  Training texts:        {meta.get('tokenizer_training_texts', 'n/a')}")
        print(f"  Phase text counts:     {json.dumps(meta.get('phase_text_counts', {}), indent=4)}")

    print()
    print("=" * 60)
    print("SUCCESS")
    print("=" * 60)
    print(f"\nTo train v9 with the new tokenizer, use:")
    print(f"  python train_curriculum.py \\")
    print(f"    --profile hybrid_medium \\")
    print(f"    --bundle-dir distillation_bundle_v4 \\")
    print(f"    --run-dir live_distill_v9 \\")
    print(f"    --build-dir C:/Users/Oxta/Desktop/reimagined-main/OXN/nsos/build-mvp/Release")
    print()
    print("Expected improvement:")
    print("  - No more C++ tokens (ensure_kv_cache_capac etc.)")
    print("  - BPE vocabulary dominated by English subwords")
    print("  - Coherent English generation at inference")


if __name__ == "__main__":
    main()
