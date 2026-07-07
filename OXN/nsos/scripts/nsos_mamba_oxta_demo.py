from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _ensure_repo_python_path() -> None:
    here = Path(__file__).resolve()
    python_dir = here.parents[1] / "python"
    sys.path.insert(0, str(python_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="NSOS Mamba standalone Oxta demo")
    parser.add_argument("--ext-dir", default="", help="directory containing nsos_ext")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--bench-tokens", type=int, default=1024)
    args = parser.parse_args()

    if args.ext_dir:
        ext_dir = str(Path(args.ext_dir).resolve())
        sys.path.insert(0, ext_dir)
        os.environ["NSOS_EXT_PATH"] = ext_dir
    _ensure_repo_python_path()

    from nsos_mamba import CharTokenizer, MambaModuleConfig, NSOSMamba, TextPair

    pairs = [
        TextPair("Quem é você?", "Oxta"),
        TextPair("quem é você?", "Oxta"),
        TextPair("Quem e voce?", "Oxta"),
    ]
    tok = CharTokenizer.from_texts([p.prompt for p in pairs] + [p.answer for p in pairs])

    cfg = MambaModuleConfig(
        vocab_size=tok.vocab_size,
        num_layers=args.layers,
        d_model=args.d_model,
        device="auto",
        streaming=True,
    )
    mamba = NSOSMamba(cfg)

    def progress(step: int, loss: float) -> None:
        print(f"[train] step {step}/{args.steps} loss~{loss:.4f}", flush=True)

    mamba.fit_text_pairs(
        pairs,
        tok,
        steps=args.steps,
        batch_size=args.batch,
        learning_rate=args.lr,
        callback=progress,
    )

    result = mamba.generate_text("Quem é você?", tok, max_new_tokens=8)
    print("\n========== RESULTADO ==========")
    print("prompt: Quem é você?")
    print("esperado: Oxta")
    print("gerado:", repr(result.text))
    print("ids:", result.generated_ids)
    print("decode tok/s:", f"{result.decode_tokens_per_sec:.2f}")
    print("e2e tok/s:", f"{result.end_to_end_tokens_per_sec:.2f}")
    print("ok:", result.text == "Oxta")

    prompt_ids = tok.encode("Quem é você?", bos=True)
    bench = mamba.benchmark_decode(prompt_ids, max_new_tokens=args.bench_tokens, repeats=5)
    bench.print()


if __name__ == "__main__":
    main()
