"""probe_cascade.py — A/B probe unificado de TODAS as tecnologias Oxta.

Treina um modelo pequeno com flags configuráveis:
  --use-chrass --chrass-density 10
  --use-ttt
  --vib-beta 0.01
  --use-slender
  --moe-expert-hidden 0|5120|...
  --use-moe

Output: probe_<label>_s<seed>.json com loss curve completo + métricas
de timing.  Usado pelo notebook cascade Colab.

Uso típico:
  python probe_cascade.py --label baseline --seed 42 --steps 1500
  python probe_cascade.py --label chrass10 --seed 42 --steps 1500 \\
      --use-chrass --chrass-density 10
  python probe_cascade.py --label full --seed 42 --steps 1500 \\
      --use-chrass --chrass-density 10 \\
      --use-ttt --vib-beta 0.01 --use-slender
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _import_nsos_ext():
    import_errors: List[str] = []
    try:
        import nsos_ext  # noqa
        return nsos_ext
    except ImportError as exc:
        import_errors.append(f"default sys.path: {exc}")
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "build-chrass-validation" / "Release",
        here.parent / "build-standalone-sm75" / "Release",
        here.parent / "build" / "Release",
        here.parent / "build-mvp" / "Release",
        here.parent / "build-colab",
        here.parent / "build-colab" / "Release",
    ]
    for p in candidates:
        if p.exists():
            sys.path.insert(0, str(p))
            try:
                import nsos_ext  # noqa
                return nsos_ext
            except ImportError as exc:
                import_errors.append(f"{p}: {exc}")
    raise ImportError(
        "nsos_ext not found. Build with NSOS_BUILD_PYTHON=ON. Attempts: "
        + " | ".join(import_errors)
    )


def build_model(eng_mod, args):
    cfg = eng_mod.ModelConfig()
    cfg.num_layers = args.layers
    cfg.d_model = args.d_model
    cfg.vocab_size = args.vocab_size
    cfg.n_heads = args.n_heads
    cfg.n_kv_heads = max(1, args.n_heads // 2)
    cfg.sliding_window = args.max_context
    cfg.attention_period = 2
    cfg.attention_slot = 1
    cfg.use_gradient_checkpointing = False
    cfg.dropout = 0.0
    cfg.max_context_tokens = args.max_context
    cfg.default_batch_size = args.batch_size
    cfg.use_cuda = args.cuda

    # Tech flags
    cfg.use_chrass = args.use_chrass
    if args.use_chrass:
        cfg.chrass_density = float(args.chrass_density) / 100.0
        cfg.chrass_seed = args.seed

    cfg.use_ttt = args.use_ttt
    if args.use_ttt:
        cfg.ttt_period = 4
        cfg.ttt_slot = 2

    cfg.use_moe = args.use_moe
    if args.use_moe:
        cfg.moe_period = 3
        cfg.moe_slot = 2
        cfg.num_experts = 4
        cfg.num_experts_per_token = 2
        cfg.moe_expert_hidden_dim = args.moe_expert_hidden

    cfg.pantheon_vib_beta = float(args.vib_beta)
    cfg.use_slender_embedding = args.use_slender

    engine = eng_mod.InferenceEngine()
    ok = engine.load_model("", cfg)
    if not ok:
        raise RuntimeError("engine.load_model returned False")
    return engine, cfg


def load_corpus(args) -> List[str]:
    if args.data_path and Path(args.data_path).exists():
        texts: List[str] = []
        path = Path(args.data_path)
        if path.suffix == ".zst":
            import zstandard as zstd
            with open(path, "rb") as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as r:
                    buf = b""
                    while True:
                        chunk = r.read(1 << 16)
                        if not chunk:
                            break
                        buf += chunk
                        while b"\n" in buf and len(texts) < args.max_docs:
                            line, _, buf = buf.partition(b"\n")
                            try:
                                row = json.loads(line.decode("utf-8"))
                                t = (row.get("text") or "").strip()
                                if len(t) >= 200:
                                    texts.append(t[:args.max_doc_chars])
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                continue
                        if len(texts) >= args.max_docs:
                            break
        else:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if len(texts) >= args.max_docs:
                        break
                    try:
                        row = json.loads(line)
                        t = (row.get("text") or "").strip()
                        if len(t) >= 200:
                            texts.append(t[:args.max_doc_chars])
                    except json.JSONDecodeError:
                        continue
        return texts
    # Synthetic PT contábil
    seeds = [
        "O ICMS sobre operacoes interestaduais de venda de mercadorias e regulado pela Lei Kandir 87/96 e suas alteracoes posteriores aplicaveis aos contribuintes do imposto sobre circulacao de mercadorias e servicos.",
        "A pessoa juridica tributada pelo lucro real apura o IRPJ pela aliquota de 15% sobre o lucro real ajustado mensalmente, com adicional de 10% sobre a parcela do lucro que exceder R$ 20.000 por mes.",
        "A Contribuicao Social sobre o Lucro Liquido CSLL tem aliquota de 9% para a maioria das empresas em regime de lucro real, exceto instituicoes financeiras que aplicam 15%.",
        "PIS e COFINS no regime nao-cumulativo tem aliquotas de 1,65% e 7,6% respectivamente sobre a receita bruta operacional, com direito a creditos sobre insumos da atividade economica.",
        "A Nota Fiscal Eletronica substitui a nota fiscal em papel e e obrigatoria para a maior parte dos contribuintes do ICMS desde a sua implementacao gradual a partir de 2008.",
        "O Simples Nacional unifica em uma so guia tributos federais, estaduais e municipais para empresas com faturamento ate R$ 4,8 milhoes anuais, simplificando a apuracao tributaria.",
        "O contador deve manter livros fiscais e contabeis pelo prazo minimo de cinco anos a contar do encerramento do exercicio social conforme legislacao tributaria federal brasileira.",
        "A apuracao do lucro real exige manutencao do Livro de Apuracao do Lucro Real LALUR para registrar ajustes ao lucro contabil, divididos entre parte A para adicoes/exclusoes e parte B para controle.",
        "Empresas optantes pelo lucro presumido aplicam percentuais de presuncao sobre a receita bruta para calculo do IRPJ (8% para comercio/industria, 32% para servicos) e CSLL (12% ou 32%).",
        "Balanco patrimonial e Demonstracao do Resultado do Exercicio DRE sao demonstracoes contabeis obrigatorias pela Lei 6404/76, complementadas pela DFC e DMPL para sociedades anonimas.",
    ]
    return (seeds * (args.max_docs // 10 + 1))[:args.max_docs]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True, help="Run identifier (becomes filename)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--log-every", type=int, default=50)
    # Model size
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--max-context", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    # Tech flags
    ap.add_argument("--use-chrass", action="store_true")
    ap.add_argument("--chrass-density", type=int, default=10)
    ap.add_argument("--use-ttt", action="store_true")
    ap.add_argument("--vib-beta", type=float, default=0.0)
    ap.add_argument("--use-slender", action="store_true")
    ap.add_argument("--use-moe", action="store_true")
    ap.add_argument("--moe-expert-hidden", type=int, default=0)
    # Data
    ap.add_argument("--data-path", type=str, default="")
    ap.add_argument("--max-docs", type=int, default=2000)
    ap.add_argument("--max-doc-chars", type=int, default=1024)
    ap.add_argument("--out-dir", type=Path, default=Path("cascade_reports"))
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Slender GPU guard ────────────────────────────────────────────────
    # Slender ensure_slender_cache_ requires weight.data on CPU. The research
    # probe therefore defines --cuda + --use-slender as an unsupported
    # combination. Detect it before loading the extension so no invalid run can
    # produce a report that resembles a completed benchmark.
    if args.use_slender and args.cuda:
        skip_report = {
            "label": args.label,
            "skipped": True,
            "reason": "Slender GPU is outside this probe's supported contract. Rerun without --cuda or without --use-slender.",
            "config": {
                "use_chrass": args.use_chrass, "use_ttt": args.use_ttt,
                "vib_beta": args.vib_beta, "use_slender": args.use_slender,
                "use_moe": args.use_moe,
            },
            "loss": {"all": [], "initial_50": float("nan"), "final_100": float("nan"), "min": float("nan")},
            "timing": {"ms_per_step": 0, "build_s": 0, "train_s": 0},
            "steps_completed": 0,
        }
        out = args.out_dir / f"probe_{args.label}_s{args.seed}.json"
        out.write_text(json.dumps(skip_report, indent=2))
        print(f"[SKIP] {args.label}: Slender + CUDA incompatible. Wrote skip-report to {out}")
        return 0

    eng_mod = _import_nsos_ext()

    flags_summary = []
    if args.use_chrass: flags_summary.append(f"chrass{args.chrass_density}")
    if args.use_ttt: flags_summary.append("ttt")
    if args.vib_beta > 0: flags_summary.append(f"vib{args.vib_beta}")
    if args.use_slender: flags_summary.append("slender")
    if args.use_moe: flags_summary.append("moe")
    if args.moe_expert_hidden > 0: flags_summary.append(f"km{args.moe_expert_hidden}")
    if not flags_summary: flags_summary.append("baseline")

    print(f"==========================================================")
    print(f" probe_cascade: {args.label} ({'+'.join(flags_summary)})")
    print(f"   seed={args.seed} steps={args.steps}")
    print(f"   model={args.layers}L d={args.d_model} h={args.n_heads} vocab={args.vocab_size}")
    print(f"==========================================================")

    texts = load_corpus(args)
    n = len(texts)
    n_eval = max(8, int(n * 0.1))
    train, eval_set = texts[:n - n_eval], texts[n - n_eval:]
    print(f"  data: {len(train)} train, {len(eval_set)} eval")

    t_build = time.time()
    engine, cfg = build_model(eng_mod, args)
    t_build = time.time() - t_build
    print(f"  built in {t_build:.2f}s, memory={engine.get_memory_usage()}")

    random.seed(args.seed)
    order = list(range(len(train)))
    random.shuffle(order)

    losses: List[float] = []
    eval_records: List[Dict[str, float]] = []
    nan_count = 0

    t_start = time.time()
    for step in range(1, args.steps + 1):
        doc = train[order[(step - 1) % len(order)]]
        try:
            loss = float(engine.train_text(doc))
        except Exception as e:
            print(f"step {step}: {e}")
            loss = float("nan")
        if loss != loss:
            nan_count += 1
            continue
        losses.append(loss)
        if step % args.log_every == 0 or step == 1:
            w = losses[-args.log_every:]
            avg = sum(w) / len(w)
            elapsed = time.time() - t_start
            eta_min = (args.steps - step) * elapsed / step / 60
            print(f"  step={step:>5d}/{args.steps}  loss={avg:.4f}  "
                  f"elapsed={elapsed:.0f}s  eta_min={eta_min:.1f}  nan={nan_count}")
        if step % args.eval_every == 0 and eval_set:
            esum, ec = 0.0, 0
            for et in eval_set[:min(8, len(eval_set))]:
                try:
                    el = float(engine.train_text(et[:args.max_doc_chars // 2]))
                    if el == el:
                        esum += el; ec += 1
                except Exception as exc:
                    raise RuntimeError(
                        f"Evaluation failed at training step {step}"
                    ) from exc
            em = esum / max(ec, 1)
            eval_records.append({"step": step, "eval_loss": em, "n": ec})
            print(f"    [eval @ {step}] eval_loss={em:.4f}")

    t_train = time.time() - t_start

    init = sum(losses[:50]) / max(len(losses[:50]), 1) if losses else float("nan")
    final = sum(losses[-100:]) / max(len(losses[-100:]), 1) if losses else float("nan")
    report = {
        "label": args.label,
        "flags": flags_summary,
        "seed": args.seed,
        "steps_target": args.steps,
        "steps_completed": len(losses),
        "nan_count": nan_count,
        "config": {
            "use_chrass": args.use_chrass,
            "chrass_density": args.chrass_density if args.use_chrass else 0,
            "use_ttt": args.use_ttt,
            "vib_beta": args.vib_beta,
            "use_slender": args.use_slender,
            "use_moe": args.use_moe,
            "moe_expert_hidden": args.moe_expert_hidden,
        },
        "model": {
            "layers": args.layers, "d_model": args.d_model,
            "n_heads": args.n_heads, "vocab_size": args.vocab_size,
        },
        "timing": {
            "build_s": t_build, "train_s": t_train,
            "ms_per_step": t_train * 1000 / max(len(losses), 1),
        },
        "loss": {
            "initial_50": init, "final_100": final,
            "min": min(losses) if losses else float("nan"),
            "all": losses,
        },
        "eval_records": eval_records,
    }
    out = args.out_dir / f"probe_{args.label}_s{args.seed}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  ✓ report: {out}")
    print(f"  loss init->final: {init:.4f} -> {final:.4f} ({(init-final)/max(init,1e-6)*100:+.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
