"""Relatório de checkpoint: telemetria + eval held-out + geração fixa.

Loss instantânea de treino é ruído de microbatch. O que informa decisão é:
métricas agregadas, loss em dados não vistos, throughput sustentado, saúde do
gradiente e como o modelo de fato escreve.

Roda em CPU por padrão para não competir com um treino em andamento na GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import train_ptbr_conversational as pt  # noqa: E402

# Mesmos prompts de `sample_step320_cpu.py`, para que as gerações sejam
# comparáveis com as amostragens históricas do projeto.
HISTORIC_PROMPTS = [
    "Hello!",
    "Can you help me with a math problem?",
    "What is the capital of France?",
    "If John has 5 apples and buys 3 more, how many apples does he have?",
    "Explain what technology is.",
    "Who discovered gravity?",
    "Write a short paragraph about science.",
]
# Os prompts históricos são em INGLÊS e o corpus é PT-BR. A partir do passo
# ~75.000 eles passaram a medir sobretudo a incapacidade do modelo em inglês —
# fragmentação em subpalavra (`Explaine`, `Exhain`, `Exchain`) — enquanto as
# saídas em português seguiam bem formadas. Medir o eixo errado.
#
# Este bloco cobre as mesmas categorias (saudação, ajuda, fato, aritmética,
# explicação, atribuição, redação) no idioma que o modelo de fato aprendeu,
# para que a degradação observada seja do MODELO e não do descasamento de
# idioma. A série histórica fica intacta ao lado, para comparabilidade.
PTBR_PROMPTS = [
    "Olá!",
    "Você pode me ajudar com um problema de matemática?",
    "Qual é a capital da França?",
    "Se João tem 5 maçãs e compra mais 3, quantas maçãs ele tem?",
    "Explique o que é tecnologia.",
    "Quem descobriu a gravidade?",
    "Escreva um parágrafo curto sobre ciência.",
]

SYSTEM_PREFIX = (
    "<|bos|><|system|>\nVocê é um assistente conversacional brasileiro. "
    "Responda somente em português do Brasil, com clareza, honestidade e "
    "naturalidade. Quando não souber algo, diga que não sabe em vez de "
    "inventar.\n<|user|>\n"
)
SYSTEM_SUFFIX = "\n<|assistant|>\n"


def latest_generation(run_dir: Path, step: int | None) -> Path:
    gens = sorted((run_dir / "checkpoints" / "generations").glob("step-*"))
    if not gens:
        raise SystemExit(f"Nenhum checkpoint em {run_dir}")
    if step is None:
        return gens[-1]
    match = [g for g in gens if f"{step:012d}" in g.name]
    if not match:
        raise SystemExit(
            f"Passo {step} não encontrado. Disponíveis: "
            + ", ".join(g.name for g in gens)
        )
    return match[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--build-dir", default="build-codex-hip")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--eval-windows", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    workspace = Path(args.workspace)
    gen = latest_generation(run_dir, args.step)
    print(f"checkpoint: {gen.name}\n")

    progress = json.loads((gen / "progress.json").read_text("utf-8"))
    manifest = json.loads((gen / "checkpoint_manifest.json").read_text("utf-8"))
    telemetry = progress.get("telemetry_snapshot", {})

    print("=" * 72)
    print("TELEMETRIA")
    print("=" * 72)
    print(f"  global_step            {progress.get('global_step'):,}")
    for scope in ("phases", "rolling", "run"):
        block = telemetry.get(scope, {})
        if scope == "phases":
            block = block.get("base", {})
        if not block:
            continue
        label = {"phases": "fase base", "rolling": "janela",
                 "run": "corrida"}[scope]
        print(f"  --- {label}")
        for key, fmt in (
            ("loss_mean", "  loss media            {:.5f}"),
            ("loss_first", "  loss primeira         {:.5f}"),
            ("loss_last", "  loss ultima           {:.5f}"),
            ("raw_tokens_per_second", "  throughput            {:.1f} tok/s"),
            ("steps_per_second", "  passos/s              {:.4f}"),
            ("elapsed_seconds", "  decorrido             {:.0f} s"),
        ):
            if block.get(key) is not None:
                print("  " + fmt.format(block[key]))

    # --- Saúde do gradiente ------------------------------------------------
    #
    # `progress.json` não persiste norma de gradiente nem clipping. Os campos
    # existem no Trainer (`last_grad_norm_pre_clip`, `last_grad_norm_post_clip`,
    # `last_update_was_clipped`) e nos bindings, mas nada os grava. Sem isso
    # não há como saber, a posteriori, se o clipping esteve ativo — e clipping
    # constante é o sintoma clássico de LR alto demais.
    print("\n" + "=" * 72)
    print("GRADIENTE / CLIPPING")
    print("=" * 72)
    health = progress.get("gradient_health")
    if isinstance(health, dict) and health.get("instrumented"):
        n = health.get("updates_observed", 0)
        if n:
            frac = health["clipped_fraction"]
            print(f"  updates na janela              {n:,}")
            print(f"  updates cortados               "
                  f"{health['clipped_updates']:,} ({100 * frac:.1f}%)")
            print(f"  norma pré-clip  média          "
                  f"{health['grad_norm_mean_pre_clip']:.4f}")
            print(f"  norma pré-clip  min / max      "
                  f"{health['grad_norm_min_pre_clip']:.4f} / "
                  f"{health['grad_norm_max_pre_clip']:.4f}")
            print(f"  último pré / pós clip          "
                  f"{health['last_pre_clip']:.4f} / "
                  f"{health['last_post_clip']:.4f}")
            # Clipping em 100% não é por si só defeito: a sonda de 2026-08-09
            # mediu a melhor loss justamente com teto 1,0 e piora monotônica
            # ao afrouxar. O que o número informa é se o teto está atuando
            # como estabilizador ou como normalização permanente.
            if frac >= 0.99:
                print("  NOTA: todo update cortado — o teto está agindo como "
                      "normalização, não como rede de segurança.")
        else:
            print("  nenhum update observado nesta janela")
    else:
        print("  NÃO INSTRUMENTADO — os campos existem no Trainer e nos")
        print("  bindings, mas o checkpoint não os persiste. Não é possível")
        print("  saber se houve clipping neste passo. Ver nota no relatório.")

    # --- Eval held-out -----------------------------------------------------
    nsos = pt.load_nsos(Path(args.build_dir))
    device = nsos.Device.GPU if args.device == "gpu" else nsos.Device.CPU
    tokenizer = nsos.Tokenizer()
    tok = gen / "tokenizer.nsos"
    if not tok.is_file():
        cand = sorted((workspace / "tokenizer").glob("*.ox3"))
        if not cand:
            raise SystemExit("Tokenizer não encontrado")
        tok = cand[0]
    tokenizer.load(str(tok))

    pack = json.loads(
        (workspace / "packs" / "pack_manifest.json").read_text("utf-8")
    )
    # `vocab_size` vem do MANIFESTO, não do tokenizer carregado. O tokenizer
    # do workspace (.ox3) reporta 16.377 — o vocabulário efetivamente
    # preenchido pelo BPE — enquanto o modelo foi construído com o alvo de
    # 16.384. Usar o primeiro monta uma ModelConfig com digest diferente e o
    # load do checkpoint é recusado com "architecture mismatch".
    vocab_size = int(
        manifest["identity"]["model_config"]["vocab_size"]
    )
    config = pt.build_model_config(
        nsos, dict(pt.PRESETS["pilot"]), vocab_size, device
    )
    model = nsos.JambaModel(config, device)
    model.to(device)
    model.load(str(gen / "model.bin"), True)

    print("\n" + "=" * 72)
    print(f"EVAL HELD-OUT (base_eval, {args.eval_windows} janelas, {args.device})")
    print("=" * 72)
    started = time.monotonic()
    try:
        shard = pack["phases"]["base_eval"]["shards"][0]["file"]
        loss = pt.causal_validation_loss(
            nsos, model, workspace / "packs" / "base_eval" / shard,
            int(pt.PRESETS["pilot"]["seq_len"]), args.eval_windows,
        )
        print(f"  eval loss              {loss:.5f}")
        print(f"  perplexidade           {pow(2.718281828, loss):.2f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FALHOU: {type(exc).__name__}: {exc}")
    print(f"  tempo                  {time.monotonic() - started:.1f} s")

    # --- Geração fixa ------------------------------------------------------
    eos = int(pack["tokenizer"]["eos_token_id"])
    import train_curriculum as tc  # noqa: E402

    def run_block(title: str, prompts: list[str]) -> None:
        print("\n" + "=" * 72)
        print(title)
        print("=" * 72)
        for prompt in prompts:
            full = SYSTEM_PREFIX + prompt + SYSTEM_SUFFIX
            try:
                out = tc.greedy_generate(
                    nsos, model, tokenizer, full, args.max_new_tokens, eos
                )
            except Exception as exc:  # noqa: BLE001
                out = f"<falhou: {type(exc).__name__}: {exc}>"
            print(f"\n  > {prompt}")
            print(f"    {out}")

    run_block("GERAÇÃO PT-BR — mesmo eixo do corpus", PTBR_PROMPTS)
    run_block(
        "GERAÇÃO — prompts históricos em inglês (sample_step320_cpu.py)",
        HISTORIC_PROMPTS,
    )

    print(f"\nidentity_sha256 {manifest.get('identity_sha256', '')[:40]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
