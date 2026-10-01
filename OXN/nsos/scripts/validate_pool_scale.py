"""
NSOS — validate_pool_scale.py

Valida em ESCALA o ganho do caching allocator de GPU (ManagedPool): roda
profile_bottlenecks.py duas vezes — pool ON (NSOS_GPU_POOL=1) e pool OFF
(NSOS_GPU_POOL=0) — e imprime a tabela de speedup por config.  Cada execução é
um PROCESSO separado de propósito: o pool é um singleton que lê NSOS_GPU_POOL na
primeira alocação de GPU, então A/B no mesmo processo não funcionaria.

Os batch sizes são escolhidos pela memória da GPU detectada (a 1050 Ti/4GB
satura em bs16; T4/16GB e A100/40GB precisam de batches maiores p/ encher a GPU
e mostrar o ganho real).  Override com --batch-sizes.

Uso (Colab T4/A100, após o colab_bootstrap construir nsos_ext em build-colab):
    python OXN/nsos/scripts/validate_pool_scale.py --profile mamba_small

Uso (local 1050 Ti):
    python OXN/nsos/scripts/validate_pool_scale.py --build-dir OXN/nsos/build-cuda-validation
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional


def detect_gpu_mem_mb() -> Optional[float]:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        total = pynvml.nvmlDeviceGetMemoryInfo(h).total / (1024.0 * 1024.0)
        name = pynvml.nvmlDeviceGetName(h)
        pynvml.nvmlShutdown()
        if isinstance(name, bytes):
            name = name.decode()
        print(f"[scale] GPU: {name}  ({total:.0f} MB)")
        return total
    except Exception:
        return None


def auto_batch_sizes(mem_mb: Optional[float]) -> List[int]:
    if mem_mb is None:
        return [16]
    if mem_mb >= 32000:      # A100 40GB / H100
        return [64, 128, 256]
    if mem_mb >= 14000:      # T4 16GB / L4 / V100
        return [32, 64, 128]
    if mem_mb >= 7000:       # 8GB cards
        return [16, 32]
    return [16]              # 4GB (1050 Ti) — saturates here


def run_profiler(pool_on: bool, args, batch_csv: str, out_json: Path) -> List[Dict]:
    env = dict(os.environ)
    env["NSOS_GPU_POOL"] = "1" if pool_on else "0"
    cmd = [
        sys.executable, str(Path(__file__).with_name("profile_bottlenecks.py")),
        "--device", "gpu", "--profile", args.profile,
        "--batch-sizes", batch_csv, "--seq-len", str(args.seq_len),
        "--steps", str(args.steps), "--warmup", str(args.warmup),
        "--configs", args.configs, "--skip-forward", "--out", str(out_json),
    ]
    if args.build_dir:
        cmd += ["--build-dir", str(args.build_dir)]
    subprocess.run(cmd, env=env, check=True)
    return json.loads(out_json.read_text(encoding="utf-8")).get("rows", [])


def key(r: Dict) -> tuple:
    return (r.get("batch_size"), r.get("config"))


def main() -> int:
    p = argparse.ArgumentParser(description="Valida o ganho do GPU pool em escala (ON vs OFF).")
    p.add_argument("--build-dir", type=Path, default=None)
    p.add_argument("--profile", default="mamba_small")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--configs", default="baseline,full")
    p.add_argument("--batch-sizes", default=None, help="CSV; auto pela memória da GPU se omitido")
    args = p.parse_args()

    mem = detect_gpu_mem_mb()
    batch_csv = args.batch_sizes or ",".join(str(b) for b in auto_batch_sizes(mem))
    print(f"[scale] batch-sizes={batch_csv} seq={args.seq_len} steps={args.steps} "
          f"profile={args.profile}\n")

    with tempfile.TemporaryDirectory() as td:
        off = {key(r): r for r in run_profiler(False, args, batch_csv, Path(td) / "off.json")}
        on = {key(r): r for r in run_profiler(True, args, batch_csv, Path(td) / "on.json")}

    print("\n================= POOL ON vs OFF (training tok/s) =================")
    print(f"{'bs':>4} {'config':<9} {'OFF tok/s':>11} {'ON tok/s':>11} {'speedup':>8} "
          f"{'util OFF->ON':>14} {'mem ON MB':>10}")
    for k in sorted(set(off) | set(on)):
        ro, rn = off.get(k, {}), on.get(k, {})
        to, tn = ro.get("tokens_per_s"), rn.get("tokens_per_s")
        spd = f"{tn / to:.2f}x" if (to and tn and to > 0) else "-"
        uo = ro.get("gpu_util_mean_pct", "?")
        un = rn.get("gpu_util_mean_pct", "?")
        mem_on = rn.get("gpu_mem_peak_mb", "?")
        bs, cfg = k
        print(f"{bs:>4} {cfg:<9} {str(to):>11} {str(tn):>11} {spd:>8} "
              f"{str(uo) + '->' + str(un):>14} {str(mem_on):>10}")
    print("==================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
