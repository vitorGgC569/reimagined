"""
NSOS — profile_bottlenecks.py  (gate da Fase 1 GPU-first)

Mede o throughput REAL de TREINO (tokens/s, tempo/step) numa matriz de flags
arquiteturais (baseline vs +MoE vs +KAN vs +SSA vs full), variando batch_size,
e — em GPU — amostra utilization.gpu/memory.used via nvidia-smi. Não há mock:
constrói um JambaModel real, roda passos reais de train_supervised_batch e
cronometra.

Uso típico (gate da Fase 1):
    python scripts/profile_bottlenecks.py --device gpu --profile mamba_small \
        --batch-sizes 16,32 --seq-len 160 --steps 8 --warmup 2

Saída: tabela no stdout + JSON em artifacts/bottleneck_profile_<ts>.json.
Reusa os helpers canônicos de train_curriculum.py (load_nsos/resolve_profile/
build_model_config) para refletir exatamente a configuração de produção.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from train_curriculum import (  # helpers canônicos — mesma config de produção
    build_model_config,
    detect_build_dir,
    load_nsos,
    resolve_profile,
    set_model_training_mode,
)

# ── Matriz de flags: cada config liga UM caminho sobre o baseline ──────────────
# (chaves de ModelConfig em `cfg`; "ssa" é ligado no modelo, não no config).
FLAG_MATRIX: Dict[str, Dict[str, Any]] = {
    "baseline":  {"cfg": {"use_moe": False, "use_kan": False}, "ssa": False},
    "+moe":      {"cfg": {"use_moe": True,  "use_kan": False}, "ssa": False},
    "+kan":      {"cfg": {"use_moe": False, "use_kan": True},  "ssa": False},
    "+ssa":      {"cfg": {"use_moe": False, "use_kan": False}, "ssa": True},
    "full":      {"cfg": {"use_moe": True,  "use_kan": True},  "ssa": True},
}


def _query_gpu_smi() -> Optional[tuple]:
    """Fallback: leitura via nvidia-smi (subprocess — caro; só se NVML faltar)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        util, mem = (p.strip() for p in out.split(","))
        return float(util), float(mem)
    except Exception:
        return None


class GpuSampler:
    """Amostra ocupação de GPU num thread de fundo durante a janela cronometrada.

    Usa NVML (pynvml) IN-PROCESS — consultas de ~microssegundos, sem spawn de
    processo — para NÃO roubar CPU da thread de treino (que roda com o GIL
    liberado); spawnar nvidia-smi a cada 50 ms contaminava o tok/s medido.
    Cai para nvidia-smi num intervalo brando (250 ms) só se o NVML faltar.
    Point-sampling entre steps subestima a ocupação; amostragem contínua dá
    média e pico reais — o sinal que importa p/ diagnosticar GPU ociosa.
    """

    def __init__(self, enabled: bool, interval: float = 0.05):
        self.enabled = enabled
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.utils: List[float] = []
        self.mems: List[float] = []
        self._nvml = None
        self._handle = None

    def _init_nvml(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml = None

    def _sample(self) -> Optional[tuple]:
        if self._nvml is not None:
            try:
                u = self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
                m = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / (1024.0 * 1024.0)
                return float(u), float(m)
            except Exception:
                return None
        return _query_gpu_smi()

    def _loop(self) -> None:
        while not self._stop.is_set():
            s = self._sample()
            if s:
                self.utils.append(s[0])
                self.mems.append(s[1])
            self._stop.wait(self.interval)

    def __enter__(self) -> "GpuSampler":
        if self.enabled:
            self._init_nvml()
            if self._nvml is None and self.interval < 0.25:
                self.interval = 0.25  # subprocess fallback: amostra menos p/ não contaminar
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def stats(self) -> Dict[str, float]:
        if not self.utils:
            return {}
        return {
            "gpu_util_mean_pct": round(statistics.fmean(self.utils), 1),
            "gpu_util_peak_pct": round(max(self.utils), 1),
            "gpu_mem_peak_mb": round(max(self.mems), 1),
            "gpu_samples": len(self.utils),
        }


def synth_batch(batch_size: int, seq_len: int, vocab: int, rng: random.Random):
    """Batch sintético determinístico (ids válidos). Para PERF, conteúdo é irrelevante."""
    lo, hi = 1, max(2, vocab - 1)
    prompt_len = max(2, seq_len - 2)
    prompt_batch = [[rng.randint(lo, hi) for _ in range(prompt_len)] for _ in range(batch_size)]
    answer_batch = [[rng.randint(lo, hi), 0] for _ in range(batch_size)]  # +EOS(0)
    return prompt_batch, answer_batch


def _timed(fn, steps: int) -> List[float]:
    times: List[float] = []
    for _ in range(max(steps, 1)):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def run_config(nsos, name: str, spec: Dict[str, Any], profile: Dict, vocab: int,
               device, device_str: str, batch_size: int, seq_len: int,
               steps: int, warmup: int, lr: float,
               measure_forward: bool = True) -> Dict[str, Any]:
    rng = random.Random(1337)
    # Config base + flags da matriz.
    cfg = build_model_config(nsos, profile, vocab, device)
    for k, v in spec["cfg"].items():
        setattr(cfg, k, v)
    cfg.default_batch_size = batch_size

    model = nsos.JambaModel(cfg, device)
    model.to(device)
    if spec.get("ssa"):
        model.set_sparse_attention(True)
    trainer = nsos.Trainer(model, lr)

    prompt_batch, answer_batch = synth_batch(batch_size, seq_len, vocab, rng)
    toks_per_step = batch_size * seq_len
    on_gpu = device_str == "gpu"

    try:
        num_params = len(model.parameters())
    except Exception:
        num_params = None

    # (1) Forward-only (eval) — isola o compute de inferência do custo de
    # backward+otimizador, localizando onde o tempo do step é gasto.  Amostra a
    # ocupação aqui também: se o forward já fica em ~util baixo, o gargalo é
    # sub-ocupação (kernels pequenos / launch overhead), não sync de backward.
    # OBS: a fase de eval aloca um working-set DIFERENTE do treino; com o pool
    # de memória ligado ela infla o cache.  --skip-forward mede só o treino
    # (representativo de produção, onde não há eval intercalado).
    fwd_times = [float("nan")]
    fwd_stats: Dict[str, float] = {}
    if measure_forward:
        set_model_training_mode(model, False)
        for _ in range(max(warmup, 0)):
            model.forward_ids_batch(prompt_batch)
        with GpuSampler(on_gpu) as fwd_sampler:
            fwd_times = _timed(lambda: model.forward_ids_batch(prompt_batch), steps)
        fwd_stats = fwd_sampler.stats()

    # (2) Step de treino supervisionado completo (forward+backward+otimizador),
    # com amostragem contínua de ocupação de GPU ao longo da janela.
    set_model_training_mode(model, True)
    for _ in range(max(warmup, 0)):
        trainer.train_supervised_batch(prompt_batch, answer_batch)
    last_loss: List[float] = [float("nan")]

    def _train_step() -> None:
        last_loss[0] = float(trainer.train_supervised_batch(prompt_batch, answer_batch))

    with GpuSampler(on_gpu) as sampler:
        step_times = _timed(_train_step, steps)

    mean_step = statistics.fmean(step_times)
    result: Dict[str, Any] = {
        "config": name,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "steps": len(step_times),
        "time_step_s": round(mean_step, 4),
        "time_step_min_s": round(min(step_times), 4),
        "tokens_per_s": round(toks_per_step / mean_step, 1) if mean_step > 0 else 0.0,
        "last_loss": round(last_loss[0], 4),
        "loss_finite": bool(last_loss[0] == last_loss[0] and abs(last_loss[0]) != float("inf")),
    }
    if measure_forward and len(fwd_times) == len(step_times):
        mean_fwd = statistics.fmean(fwd_times)
        result["fwd_time_s"] = round(mean_fwd, 4)
        result["fwd_tokens_per_s"] = round(toks_per_step / mean_fwd, 1) if mean_fwd > 0 else 0.0
    if num_params is not None:
        result["num_param_tensors"] = num_params
    if fwd_stats:
        result["fwd_gpu_util_mean_pct"] = fwd_stats["gpu_util_mean_pct"]
        result["fwd_gpu_util_peak_pct"] = fwd_stats["gpu_util_peak_pct"]
    result.update(sampler.stats())
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NSOS GPU training bottleneck profiler (Fase 1 gate).")
    p.add_argument("--build-dir", type=Path, default=None)
    p.add_argument("--profile", default="mamba_small")
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--batch-sizes", default="16,32")
    p.add_argument("--seq-len", type=int, default=160)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--configs", default="baseline,+moe,+kan,+ssa,full")
    p.add_argument("--skip-forward", action="store_true",
                   help="Não mede forward-only (evita inflar o pool de memória com "
                        "o working-set de eval; mede só o treino, como em produção).")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    build_dir = detect_build_dir(args.build_dir)
    nsos = load_nsos(build_dir)
    _, profile = resolve_profile(args.profile)
    vocab = int(profile.get("target_vocab", profile.get("vocab_size", 4096)))
    lr = float(profile.get("lr", 3e-4))
    device = nsos.Device.GPU if args.device == "gpu" else nsos.Device.CPU
    batch_sizes = [int(b) for b in str(args.batch_sizes).split(",") if b.strip()]
    config_names = [c.strip() for c in str(args.configs).split(",") if c.strip() in FLAG_MATRIX]

    rows: List[Dict[str, Any]] = []
    for bs in batch_sizes:
        for name in config_names:
            try:
                r = run_config(nsos, name, FLAG_MATRIX[name], profile, vocab, device,
                               args.device, bs, args.seq_len, args.steps, args.warmup, lr,
                               measure_forward=not args.skip_forward)
            except Exception as exc:  # OOM / config inválida → registra e segue
                r = {"config": name, "batch_size": bs, "seq_len": args.seq_len,
                     "error": f"{type(exc).__name__}: {exc}"}
            rows.append(r)
            if "error" in r:
                tag = r["error"]
            else:
                fwd_tps = r.get("fwd_tokens_per_s")
                fwd_part = f" (fwd {fwd_tps:>7})" if fwd_tps is not None else ""
                tag = f"{r['tokens_per_s']:>7} tok/s{fwd_part} | {r['time_step_s']}s/step"
                if "gpu_util_mean_pct" in r:
                    fwd_u = r.get("fwd_gpu_util_mean_pct", "?")
                    tag += (f" | gpu {r['gpu_util_mean_pct']}%avg (fwd {fwd_u}%) "
                            f"{r['gpu_mem_peak_mb']}MB")
                if "num_param_tensors" in r:
                    tag += f" | {r['num_param_tensors']}p"
                if "last_loss" in r:
                    tag += f" | loss {r['last_loss']}{'' if r.get('loss_finite', True) else ' !NONFINITE!'}"
            print(f"[{args.device}] bs={bs:>3} {name:<9} -> {tag}")

    report = {
        "device": args.device, "profile": args.profile, "build_dir": str(build_dir),
        "seq_len": args.seq_len, "steps": args.steps, "vocab": vocab,
        "timestamp": datetime.now().isoformat(timespec="seconds"), "rows": rows,
    }
    out = args.out or (Path(__file__).resolve().parents[1] / "artifacts"
                       / f"bottleneck_profile_{datetime.now():%Y%m%d_%H%M%S}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nartefato: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
