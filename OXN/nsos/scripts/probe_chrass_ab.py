"""probe_chrass_ab.py — A/B probe controlado de CHRASS density.

Treina um modelo pequeno (configurable) por N steps com CHRASS desligado
ou ligado em densidade específica, mantendo seed/dados/lr idênticos entre
runs. Loga loss curve + final eval pra comparação posterior.

Objetivo: medir se CHRASS HELP, HURT, ou NEUTRO em diferentes densidades
ANTES de comprometer 6 dias de treino real na 2080 Ti do amigo.

Uso (Colab T4):

    # baseline
    python probe_chrass_ab.py --density 0 --seed 42 --steps 2000

    # densidades CHRASS
    for d in 10 20 30 40 50 60 70; do
        python probe_chrass_ab.py --density $d --seed 42 --steps 2000
    done

    # depois rodar analyze_chrass_ab.py pra agregar e plotar
    python analyze_chrass_ab.py --reports-dir probe_reports/

Modelo pequeno por default (~5M params, ~30min em T4).  Para 40M, passar
--layers 12 --d-model 384 (consome ~60min em T4).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─── Robust nsos_ext import (handles dev tree, build dir, installed) ────────
def _import_nsos_ext():
    try:
        import nsos_ext  # noqa: F401
        return nsos_ext
    except ImportError:
        pass
    # Search common build dirs
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "build-chrass-validation" / "Release",
        here.parent / "build-standalone-sm75" / "Release",
        here.parent / "build" / "Release",
        here.parent / "build-mvp" / "Release",
        here.parent.parent / "build" / "Release",
    ]
    for p in candidates:
        if p.exists() and any(f.suffix in (".pyd", ".so") for f in p.iterdir() if f.is_file()):
            sys.path.insert(0, str(p))
            try:
                import nsos_ext  # noqa
                return nsos_ext
            except ImportError:
                continue
    raise ImportError(
        "Could not find nsos_ext module. Build it first or set PYTHONPATH "
        "to the build output directory."
    )


def build_model(eng_mod, args) -> Tuple[Any, Any]:
    """Build a JambaModel via InferenceEngine.load_model('', cfg)."""
    cfg = eng_mod.ModelConfig()
    cfg.num_layers = args.layers
    cfg.d_model = args.d_model
    cfg.vocab_size = args.vocab_size
    cfg.n_heads = args.n_heads
    cfg.n_kv_heads = max(1, args.n_heads // 2)
    cfg.sliding_window = args.max_context
    cfg.attention_period = 2
    cfg.attention_slot = 1
    cfg.use_moe = False
    cfg.use_ttt = False
    cfg.use_gradient_checkpointing = False
    cfg.dropout = 0.0
    cfg.max_context_tokens = args.max_context
    cfg.default_batch_size = args.batch_size
    cfg.use_cuda = args.cuda
    # CHRASS toggle
    cfg.use_chrass = (args.density > 0)
    cfg.chrass_density = float(args.density) / 100.0
    cfg.chrass_seed = args.seed  # reuse same seed for reproducibility

    engine = eng_mod.InferenceEngine()
    ok = engine.load_model("", cfg)
    if not ok:
        raise RuntimeError("engine.load_model returned False")
    return engine, cfg


def load_training_texts(path: Path, max_chars_per_doc: int, max_docs: int) -> List[str]:
    """Read a single .jsonl (or .jsonl.zst) into a list of texts."""
    texts: List[str] = []
    if path.suffix == ".zst":
        try:
            import zstandard as zstd
        except ImportError:
            raise SystemExit("zstandard required for .zst files: pip install zstandard")
        with open(path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                buf = b""
                while True:
                    chunk = reader.read(1 << 16)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf and len(texts) < max_docs:
                        line, _, buf = buf.partition(b"\n")
                        try:
                            row = json.loads(line.decode("utf-8"))
                            t = (row.get("text") or "").strip()
                            if len(t) >= 200:
                                texts.append(t[:max_chars_per_doc])
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue
                    if len(texts) >= max_docs:
                        break
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if len(texts) >= max_docs:
                    break
                try:
                    row = json.loads(line)
                    t = (row.get("text") or "").strip()
                    if len(t) >= 200:
                        texts.append(t[:max_chars_per_doc])
                except json.JSONDecodeError:
                    continue
    return texts


def split_train_eval(texts: List[str], eval_ratio: float = 0.1) -> Tuple[List[str], List[str]]:
    """Deterministic split: last `eval_ratio` fraction goes to eval."""
    n = len(texts)
    n_eval = max(8, int(n * eval_ratio))
    return texts[:n - n_eval], texts[n - n_eval:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--density", type=int, required=True,
                    help="CHRASS density in percent: 0 (off), 10, 20, ..., 70")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--max-context", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-doc-chars", type=int, default=1024)
    ap.add_argument("--max-docs", type=int, default=2000)
    ap.add_argument("--data-path", type=Path, default=None,
                    help="Path to .jsonl or .jsonl.zst with {'text': ...} per line. "
                         "If absent, uses tiny synthetic corpus.")
    ap.add_argument("--out-dir", type=Path, default=Path("probe_reports"))
    ap.add_argument("--cuda", action="store_true",
                    help="Use CUDA if nsos_ext was built with USE_CUDA")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    eng_mod = _import_nsos_ext()

    print(f"== CHRASS A/B probe ==")
    print(f"  density: {args.density}%")
    print(f"  seed:    {args.seed}")
    print(f"  steps:   {args.steps}")
    print(f"  model:   {args.layers}L × d={args.d_model} × {args.n_heads}H")
    print(f"  cuda:    {args.cuda}")

    # ── Data
    if args.data_path is not None and args.data_path.exists():
        all_texts = load_training_texts(args.data_path, args.max_doc_chars, args.max_docs)
        print(f"  loaded {len(all_texts)} docs from {args.data_path}")
    else:
        # Synthetic corpus: deterministic, in-domain (PT contábil-ish)
        seeds = [
            "O ICMS sobre operacoes interestaduais de venda de mercadorias e regulado pela Lei Kandir 87/96 e suas alteracoes posteriores.",
            "A pessoa juridica tributada pelo lucro real apura o IRPJ pela aliquota de 15% sobre o lucro real ajustado mensalmente.",
            "A Contribuicao Social sobre o Lucro Liquido CSLL tem aliquota de 9% para a maioria das empresas em regime de lucro real.",
            "PIS e COFINS no regime nao-cumulativo tem aliquotas de 1,65% e 7,6% respectivamente sobre a receita bruta operacional.",
            "A Nota Fiscal Eletronica substitui a nota fiscal em papel e e obrigatoria para a maior parte dos contribuintes do ICMS.",
            "O Simples Nacional unifica em uma so guia tributos federais, estaduais e municipais para empresas com faturamento limitado.",
            "O contador deve manter livros fiscais e contabeis pelo prazo minimo de cinco anos a contar do encerramento do exercicio.",
            "A apuracao do lucro real exige manutencao do Livro de Apuracao do Lucro Real LALUR para registrar ajustes ao lucro contabil.",
            "Empresas optantes pelo lucro presumido aplicam percentuais de presuncao sobre a receita bruta para calculo do IRPJ e CSLL.",
            "Balanco patrimonial e Demonstracao do Resultado do Exercicio DRE sao demonstracoes contabeis obrigatorias pela Lei 6404/76.",
        ]
        # Replicate to get enough samples
        all_texts = (seeds * 200)[:args.max_docs]
        print(f"  using {len(all_texts)} synthetic PT contabil docs")

    train_texts, eval_texts = split_train_eval(all_texts, eval_ratio=0.1)
    print(f"  train={len(train_texts)} eval={len(eval_texts)}")

    # ── Build model
    print(f"  building model...")
    t_build_start = time.time()
    engine, cfg = build_model(eng_mod, args)
    t_build = time.time() - t_build_start
    mem = engine.get_memory_usage()
    print(f"  built in {t_build:.2f}s, memory={mem}")

    # ── Train loop
    import random
    random.seed(args.seed)
    doc_order = list(range(len(train_texts)))
    random.shuffle(doc_order)

    losses: List[float] = []
    log_records: List[Dict[str, float]] = []
    eval_records: List[Dict[str, float]] = []

    print(f"\n  starting {args.steps} training steps...")
    t_train_start = time.time()
    nan_count = 0
    for step in range(1, args.steps + 1):
        doc = train_texts[doc_order[(step - 1) % len(doc_order)]]
        try:
            loss = float(engine.train_text(doc))
        except Exception as exc:
            print(f"step {step}: train_text exception: {exc}")
            loss = float("nan")
        if loss != loss:  # NaN
            nan_count += 1
            continue
        losses.append(loss)

        if step % args.log_every == 0 or step == 1:
            window = losses[-args.log_every:] if len(losses) > args.log_every else losses
            avg = sum(window) / len(window)
            elapsed = time.time() - t_train_start
            eta_h = (args.steps - step) * (elapsed / step) / 3600.0
            print(f"  step={step:>5d}/{args.steps}  loss={avg:.4f}  "
                  f"elapsed={elapsed:.0f}s  eta={eta_h:.2f}h  nan={nan_count}")
            log_records.append({
                "step": step,
                "loss_window_mean": avg,
                "elapsed_s": elapsed,
                "nan_count": nan_count,
            })

        if step % args.eval_every == 0:
            # Eval loss = mean train_text on held-out (in training mode!
            # We don't have a proper eval API; use train_text on eval set
            # for the loss value but accept it perturbs weights minimally).
            # Better: skip and rely on training loss alone, OR use train_text
            # then reset to a checkpoint.  Pragmatic: log eval-on-train.
            eval_loss_sum = 0.0
            eval_count = 0
            for et in eval_texts[: min(8, len(eval_texts))]:
                try:
                    el = float(engine.train_text(et[:args.max_doc_chars // 2]))
                    if el == el:  # not NaN
                        eval_loss_sum += el
                        eval_count += 1
                except Exception:
                    pass
            eval_mean = eval_loss_sum / max(eval_count, 1)
            eval_records.append({
                "step": step,
                "eval_loss_mean": eval_mean,
                "eval_count": eval_count,
            })
            print(f"     [eval @ step={step}] eval_loss_mean={eval_mean:.4f} n={eval_count}")

    t_train = time.time() - t_train_start

    # ── Final report
    final_loss = (sum(losses[-100:]) / max(len(losses[-100:]), 1)) if losses else float("nan")
    initial_loss = (sum(losses[:50]) / max(len(losses[:50]), 1)) if losses else float("nan")
    loss_min = min(losses) if losses else float("nan")
    report = {
        "density": args.density,
        "seed": args.seed,
        "steps_target": args.steps,
        "steps_completed": len(losses),
        "nan_count": nan_count,
        "model": {
            "layers": args.layers,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "vocab_size": args.vocab_size,
            "max_context": args.max_context,
        },
        "timing": {
            "build_s": t_build,
            "train_s": t_train,
            "ms_per_step": (t_train * 1000.0 / max(len(losses), 1)),
        },
        "loss": {
            "initial_50": initial_loss,
            "final_100": final_loss,
            "min": loss_min,
            "all": losses,
        },
        "log_records": log_records,
        "eval_records": eval_records,
    }
    out_path = args.out_dir / f"probe_chrass_d{args.density:02d}_s{args.seed}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n  report -> {out_path}")
    print(f"  initial loss (mean of first 50): {initial_loss:.4f}")
    print(f"  final loss (mean of last 100):   {final_loss:.4f}")
    print(f"  min loss:                         {loss_min:.4f}")
    print(f"  loss reduction:                   {((initial_loss - final_loss) / max(initial_loss, 1e-6) * 100):.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
