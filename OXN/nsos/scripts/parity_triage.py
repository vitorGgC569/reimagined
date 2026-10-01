"""
parity_triage.py — bissecao do NaN da paridade por kill-switches.

Roda attn_bwd_parity.py em 5 configuracoes e aponta QUAL mudanca da noite
quebrou o harness FP32 (NaN/explosao nos grads):
  base      : tudo ligado (reproduz o problema)
  fused=0   : NSOS_FUSED_OPT=0      (otimizador fundido OFF -> caminho antigo)
  uninit=0  : NSOS_UNINIT=0         (elisao de zero-fill OFF -> zera tudo)
  async=0   : NSOS_ASYNC_D2D=0      (copias D2D sincronas de novo)
  all-off   : os tres OFF           (sanidade: deve ficar limpo)

Uso: python parity_triage.py --build-dir DIR
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

CONFIGS = [
    ("base",     {}),
    ("fused=0",  {"NSOS_FUSED_OPT": "0"}),
    ("uninit=0", {"NSOS_UNINIT": "0"}),
    ("async=0",  {"NSOS_ASYNC_D2D": "0"}),
    ("all-off",  {"NSOS_FUSED_OPT": "0", "NSOS_UNINIT": "0", "NSOS_ASYNC_D2D": "0"}),
]


def run_one(name, extra_env, build_dir):
    env = dict(os.environ)
    env.update(extra_env)
    script = Path(__file__).resolve().parent / "attn_bwd_parity.py"
    r = subprocess.run([sys.executable, str(script), "--build-dir", str(build_dir)],
                       env=env, capture_output=True, text=True, timeout=1800)
    out = r.stdout
    floors = re.search(r"host-vs-host = ([^\s]+)\s+gpu-vs-gpu = ([^\s]+)", out)
    effect = re.search(r"host-vs-gpu\s+= ([^\s]+)", out)
    hh = floors.group(1) if floors else "?"
    gg = floors.group(2) if floors else "?"
    ef = effect.group(1) if effect else "?"
    sick = ("nan" in out.lower().split("perfil")[0] or
            any(_big(v) for v in (hh, gg, ef)))
    verdict = "DOENTE" if sick else "limpo"
    print(f"[triage] {name:<9} host-vs-host={hh:<12} gpu-vs-gpu={gg:<12} "
          f"efeito={ef:<12} -> {verdict}")
    return not sick


def _big(v):
    try:
        x = float(v)
        return x != x or x > 10.0  # nan ou explosao
    except ValueError:
        return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", required=True)
    args = ap.parse_args()
    results = {}
    for name, env in CONFIGS:
        try:
            results[name] = run_one(name, env, args.build_dir)
        except Exception as exc:
            print(f"[triage] {name}: ERRO {exc}")
            results[name] = False
    print("-" * 64)
    if results.get("base"):
        print("[triage] base LIMPO — problema nao reproduziu nesta sessao")
        return 0
    culprits = [n for n in ("fused=0", "uninit=0", "async=0") if results.get(n)]
    if culprits:
        print(f"[triage] CULPADO(S): a(s) mudanca(s) desligada(s) em {culprits} "
              f"— desligar ESSA flag conserta o harness")
    elif results.get("all-off"):
        print("[triage] nenhuma flag isolada conserta, mas all-off sim — interacao entre mudancas")
    else:
        print("[triage] nem all-off conserta — causa anterior aos batches da noite; investigar fora deles")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
