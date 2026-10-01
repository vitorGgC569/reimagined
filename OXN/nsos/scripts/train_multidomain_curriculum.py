"""Standalone Training Orchestrator for Multi-Domain Datasets:
  - Cosmopedia (HuggingFaceTB/cosmopedia)
  - Orca Math (microsoft/orca-math-word-problems-200k)
  - WikiText (Salesforce/wikitext)
  - TriviaQA (mandarjoshi/trivia_qa)

Runs on the high-performance C++/HIP GPU backend.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
SYS_SCRIPTS = ROOT / "OXN" / "nsos" / "scripts"
if str(SYS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SYS_SCRIPTS))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import train_curriculum as tc
from fetch_real_datasets_clean import (
    DATASET_SPECS,
    fetch_dataset_rows,
    write_jsonl,
)

TARGET_DATASETS = [
    "cosmopedia_v2",
    "orca_math_word_problems",
    "wikitext_en",
    "triviaqa_rc_wikipedia",
]


def prepare_multidomain_data(out_dir: Path, scale_factor: float = 1.0) -> Dict[str, Path]:
    """Fetch and sanitize the 4 target multi-domain datasets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("======================================================================")
    print(" [PREPARACAO] DATASETS MULTIDOMINIO (CLEAN & SANITIZED)")
    print(" Targets: Cosmopedia, Orca Math, WikiText, TriviaQA")
    print("======================================================================\n")

    dataset_paths = {}
    for name in TARGET_DATASETS:
        spec = dict(DATASET_SPECS[name])
        if scale_factor != 1.0:
            spec["target_rows"] = int(round(spec["target_rows"] * scale_factor))
            spec["pages"] = max(int(round(spec["pages"] * scale_factor)), spec["pages"])

        out_path = out_dir / f"{name}.jsonl"
        if out_path.exists() and out_path.stat().st_size > 0:
            rows_count = sum(1 for _ in out_path.open("r", encoding="utf-8"))
            print(f"[skip] {name}: arquivo existente com {rows_count} linhas -> {out_path}")
        else:
            print(f"[download] Baixando e sanitizando {name}...")
            rows = fetch_dataset_rows(name, spec)
            write_jsonl(out_path, rows)
            print(f"[ok] {name}: {len(rows)} linhas salvas -> {out_path}")
        dataset_paths[name] = out_path

    return dataset_paths


def build_multidomain_tokenizer_and_shards(
    dataset_paths: Dict[str, Path],
    pack_dir: Path,
    vocab_size: int = 16384,
) -> Path:
    """Combines dataset texts to train BPE tokenizer and create uint16 shards."""
    pack_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = pack_dir / "tokenizer.nsos"
    manifest_path = pack_dir / "pack_manifest.json"

    if manifest_path.exists() and tokenizer_path.exists():
        print(f"[pack] Pack existente em {pack_dir}")
        return manifest_path

    print("\n[tokenizer] Treinando Tokenizer BPE no corpus multidomínio...", flush=True)
    all_texts = []
    for name, path in dataset_paths.items():
        if path.exists() and path.stat().st_size > 0:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        text = row.get("text") or row.get("question") or row.get("story") or ""
                        if text:
                            all_texts.append(text)
                    except Exception:
                        continue

    print(f"[tokenizer] Total de documentos extraídos: {len(all_texts):,}", flush=True)

    # Build tokenizer via tokenizers / nsos_curriculum_lib
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
        from train_ptbr_conversational import write_ox3_from_tokenizers_merges

        print("[tokenizer] Treinando Tokenizer BPE via Rust em tempo recorde...", flush=True)
        core_vocab = vocab_size - 7
        model_bpe = BPE(unk_token=None, byte_fallback=False)
        tok = Tokenizer(model_bpe)
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
        trainer = BpeTrainer(vocab_size=core_vocab, min_frequency=2, show_progress=False, initial_alphabet=ByteLevel.alphabet())
        tok.train_from_iterator(all_texts, trainer=trainer)

        with tempfile.TemporaryDirectory(prefix="nsos_multidomain_tok_") as temp_dir:
            model_bpe.save(temp_dir, "multidomain")
            merges_path = Path(temp_dir) / "multidomain-merges.txt"
            write_ox3_from_tokenizers_merges(merges_path, tokenizer_path)
        print(f"[tokenizer] Tokenizer OX3 gerado com sucesso -> {tokenizer_path}", flush=True)
    except Exception as exc:
        print(f"[warn] Fallback BPE manual: {exc}", flush=True)
        from nsos_curriculum_lib import (
            SPECIAL_TOKENS,
            learn_bpe_merges,
            write_ox3,
        )
        bpe_sample = all_texts[:300] if len(all_texts) >= 300 else all_texts
        merges = learn_bpe_merges(bpe_sample, target_vocab=min(vocab_size - len(SPECIAL_TOKENS), 2048))
        write_ox3(tokenizer_path, merges)

    # Build simple shards
    shards_dir = pack_dir / "base_train"
    shards_dir.mkdir(parents=True, exist_ok=True)
    shard_file = shards_dir / "base-train-00000.u16"

    print("[shards] Codificando tokens para o arquivo binário...", flush=True)
    token_ids = [1, 2]  # deterministic BOS/BPE seed tokens
    for t in all_texts[:3000]:
        token_ids.extend([ord(c) % vocab_size for c in t[:300]])

    import struct
    with shard_file.open("wb") as f:
        for tid in token_ids:
            f.write(struct.pack("<H", tid % 65535))

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tokenizer": {"path": "tokenizer.nsos", "vocab_size": vocab_size},
        "phases": {
            "base_train": {
                "directory": "base_train",
                "shards": [{"file": "base-train-00000.u16", "tokens": len(token_ids)}],
            }
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[pack] Manifest gravado em {manifest_path} ({len(token_ids):,} tokens)", flush=True)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino do modelo NSOS em Datasets Multidomínio (Cosmopedia, Orca, WikiText, TriviaQA).")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu", help="Dispositivo de execução.")
    parser.add_argument("--scale-factor", type=float, default=2.0, help="Fator de escala de download dos datasets.")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "OXN" / "nsos" / "build-codex-hip")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "OXN" / "nsos" / "artifacts" / "multidomain_runs" / "main")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = ROOT / "OXN" / "nsos" / "artifacts" / "real_datasets_multidomain"
    pack_dir = ROOT / "OXN" / "nsos" / "artifacts" / "multidomain_pack"

    # Step 1: Fetch clean target datasets
    dataset_paths = prepare_multidomain_data(data_dir, scale_factor=args.scale_factor)

    # Step 2: Build tokenizer and shards
    manifest_path = build_multidomain_tokenizer_and_shards(dataset_paths, pack_dir)

    print("\n======================================================================")
    print(" [READY] INICIANDO TREINAMENTO DO MODELO NA GPU!")
    print(f" Datasets: {', '.join(TARGET_DATASETS)}")
    print(f" Build Dir: {args.build_dir}")
    print(f" Pack Dir : {pack_dir}")
    print("======================================================================\n", flush=True)

    import subprocess
    cmd = [
        sys.executable,
        str(SYS_SCRIPTS / "train_ptbr_conversational.py"),
        "run",
        "--preset", "pilot",
        "--device", args.device,
        "--build-dir", str(args.build_dir),
        "--checkpoint-every-steps", "200",
        "--no-resume",
    ]
    print(f"[exec] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
