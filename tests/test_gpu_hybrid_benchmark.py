"""
tests/test_gpu_hybrid_benchmark.py

Official NSOS GPU Hybrid Architecture Benchmark on Dedicated AMD Radeon RX 7600 (HIP).
Evaluates the 4 Hybrid Structures (Excluding Pure Baselines) under Calibrated Steps:
  1. Mamba-2 Híbrido (Attn): Mamba-2 SSD + GQA Attention (lr=0.015)
  2. Mamba-3 Híbrido (Attn): Mamba-3 SISO/MIMO + GQA Attention (lr=0.002)
  3. Mamba-3 + Attn + KAN:   Mamba-3 + GQA Attention + BitFastKAN RBF splines (lr=0.002)
  4. Mamba-3 + Attn + MoE:   Mamba-3 + GQA Attention + Sparse MoE 4-expert router (lr=0.002)

Tasks Evaluated:
  1. Flappy Bird Autopilot (80 GPU trajectories, 10 live flight trials)
  2. Associative Recall / Needle-in-a-Haystack (50 train, 30 test pairs on GPU)
  3. Algorithmic Sequence Reversal (40 train, 30 test sequences on GPU)

Hardware:
  - AMD Radeon RX 7600 (8 GB VRAM, RDNA 3, gfx1102)
  - Native HIP ROCm backend via OXN/nsos/build-gm-hip
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
HIP_BUILD_DIR = REPO_ROOT / "OXN" / "nsos" / "build-gm-hip"
if not HIP_BUILD_DIR.exists() or not list(HIP_BUILD_DIR.glob("nsos_ext*.pyd")):
    raise RuntimeError(f"Could not find compiled nsos_ext in {HIP_BUILD_DIR}")

sys.path.insert(0, str(HIP_BUILD_DIR))
import nsos_ext
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    except Exception:
        pass


# -----------------------------------------------------------------------------
# 1. Environment & Tokenizers
# -----------------------------------------------------------------------------
class Pipe:
    def __init__(self, x: float, top_height: float, gap: float = 110.0, width: float = 50.0):
        self.x = x
        self.top_height = top_height
        self.gap = gap
        self.width = width
        self.passed = False

    @property
    def bottom_y(self) -> float:
        return self.top_height + self.gap

    @property
    def gap_center(self) -> float:
        return self.top_height + (self.gap / 2.0)


class FlappyEnvironment:
    WIDTH = 300
    HEIGHT = 400
    GRAVITY = 0.9
    FLAP_IMPULSE = -7.5
    MAX_VELOCITY = 9.0
    PIPE_SPEED = 3.0
    PIPE_SPAWN_DIST = 140.0
    BIRD_X = 50.0
    BIRD_RADIUS = 12.0

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.bird_y = 200.0
        self.bird_vel = 0.0
        self.pipes: List[Pipe] = []
        self.score = 0
        self.frames = 0
        self.game_over = False
        self.reset()

    def reset(self):
        self.bird_y = float(self.HEIGHT // 2)
        self.bird_vel = 0.0
        self.pipes = [
            Pipe(x=200.0, top_height=120.0, gap=110.0),
            Pipe(x=340.0, top_height=160.0, gap=110.0),
            Pipe(x=480.0, top_height=100.0, gap=110.0),
        ]
        self.score = 0
        self.frames = 0
        self.game_over = False
        return self.get_observation()

    def _spawn_pipe(self, x: float):
        top = self.rng.uniform(50.0, self.HEIGHT - 170.0)
        self.pipes.append(Pipe(x=x, top_height=top, gap=110.0))

    def get_next_pipe(self) -> Pipe:
        for p in self.pipes:
            if p.x + p.width >= self.BIRD_X - self.BIRD_RADIUS:
                return p
        return self.pipes[0]

    def get_observation(self) -> Tuple[float, float, float]:
        p = self.get_next_pipe()
        dx = (p.x + p.width / 2.0) - self.BIRD_X
        dy = p.gap_center - self.bird_y
        return dx, dy, self.bird_vel

    def step(self, action: int) -> Tuple[Tuple[float, float, float], float, bool, int]:
        if self.game_over:
            return self.get_observation(), 0.0, True, self.score

        self.frames += 1

        if action == 1:
            self.bird_vel = self.FLAP_IMPULSE

        self.bird_vel = min(self.bird_vel + self.GRAVITY, self.MAX_VELOCITY)
        self.bird_y += self.bird_vel

        for p in self.pipes:
            p.x -= self.PIPE_SPEED
            if not p.passed and p.x + p.width < self.BIRD_X:
                p.passed = True
                self.score += 1

        if self.pipes and self.pipes[0].x + self.pipes[0].width < 0:
            self.pipes.pop(0)
            last_x = self.pipes[-1].x
            self._spawn_pipe(last_x + self.PIPE_SPAWN_DIST)

        if self.bird_y - self.BIRD_RADIUS <= 0 or self.bird_y + self.BIRD_RADIUS >= self.HEIGHT:
            self.game_over = True

        next_pipe = self.get_next_pipe()
        if (next_pipe.x <= self.BIRD_X + self.BIRD_RADIUS and
            next_pipe.x + next_pipe.width >= self.BIRD_X - self.BIRD_RADIUS):
            if (self.bird_y - self.BIRD_RADIUS < next_pipe.top_height or
                self.bird_y + self.BIRD_RADIUS > next_pipe.bottom_y):
                self.game_over = True

        reward = 1.0
        if self.game_over:
            reward = -20.0
        elif action == 1 and abs(self.bird_y - next_pipe.gap_center) < 30.0:
            reward += 0.5

        return self.get_observation(), reward, self.game_over, self.score


class FlappyTokenizer:
    VOCAB_SIZE = 64
    BOS = 1
    SEP = 2
    DX_BASE = 10
    DY_BASE = 25
    VEL_BASE = 48
    ACTION_COAST = 60
    ACTION_FLAP = 61

    @classmethod
    def discretize_dx(cls, dx: float) -> int:
        b = int(max(0.0, min(200.0, dx)) / 20.0)
        return cls.DX_BASE + min(9, b)

    @classmethod
    def discretize_dy(cls, dy: float) -> int:
        norm = (dy + 150.0) / 300.0
        b = int(max(0.0, min(1.0, norm)) * 19.99)
        return cls.DY_BASE + b

    @classmethod
    def discretize_vel(cls, vel: float) -> int:
        norm = (vel + 8.0) / 18.0
        b = int(max(0.0, min(1.0, norm)) * 9.99)
        return cls.VEL_BASE + b

    @classmethod
    def encode_state(cls, obs: Tuple[float, float, float]) -> List[int]:
        dx, dy, vel = obs
        return [
            cls.BOS,
            cls.discretize_dx(dx),
            cls.discretize_dy(dy),
            cls.discretize_vel(vel),
            cls.SEP,
        ]


def flappy_expert_policy(obs: Tuple[float, float, float]) -> int:
    dx, dy, vel = obs
    target_lead = vel * 3.5
    predicted_dy = dy - target_lead
    if predicted_dy > 12.0:
        return 1
    elif predicted_dy < -20.0:
        return 0
    else:
        return 1 if vel > 2.0 else 0


class NSOSAgent:
    def __init__(self, model: nsos_ext.JambaModel):
        self.model = model
        self.model.set_training_mode(False)

    def reset_episode(self):
        self.model.reset_session()

    def predict_action(self, obs: Tuple[float, float, float]) -> int:
        prompt = FlappyTokenizer.encode_state(obs)
        logits_tensor = self.model.forward_ids(prompt, None)
        logits_np = logits_tensor.numpy()[-1]

        coast_score = float(logits_np[FlappyTokenizer.ACTION_COAST])
        flap_score = float(logits_np[FlappyTokenizer.ACTION_FLAP])

        return 1 if flap_score > coast_score else 0



# -----------------------------------------------------------------------------
# 2. GPU Hybrid Model Factory
# -----------------------------------------------------------------------------
def build_gpu_model(
    name: str,
    mamba3: bool,
    kan: bool,
    moe: bool = False,
    all_layers: bool = False,
    num_layers: int = 2,
    d_model: int = 128,
    vocab_size: int = 64,
) -> Tuple[nsos_ext.JambaModel, int]:
    # Mamba-3: usa o kernel de scan paralelo (70x mais rapido que dense_reference).
    # Mamba-2: mantem dense_reference (usa o seu proprio caminho SSD ja otimizado).
    if mamba3 or all_layers:
        os.environ["NSOS_MAMBA3_GPU_PROVIDER"] = "parallel_fp32_v1"
    else:
        os.environ["NSOS_MAMBA3_GPU_PROVIDER"] = "dense_reference"
    os.environ.setdefault("NSOS_MAMBA3_PROJECTION_PROVIDER", "exact_fp32")
    cfg = nsos_ext.ModelConfig()
    cfg.architecture_schema_version = 3
    cfg.vocab_size = vocab_size

    if all_layers:
        # Arquitetura completa: Mamba-3 + Atencao + MoE + KAN simultaneamente
        # 4 camadas com periodos separados:
        #   L1: Mamba-3 + MoE (period=2, slot=0)
        #   L2: Mamba-3 + KAN (use_kan=True, sem MoE)
        #   L3: Mamba-3 + MoE (period=2, slot=0)
        #   L4: Mamba-3 + Atencao (period=4, slot=3) + KAN
        cfg.num_layers       = 4
        cfg.d_model          = d_model
        cfg.n_heads          = 4
        cfg.n_kv_heads       = 4
        cfg.mamba3_enabled   = True
        cfg.mamba3_schema_version = 1
        cfg.attention_period = 4
        cfg.attention_slot   = 3
        cfg.use_moe              = True
        cfg.num_experts          = 4
        cfg.num_experts_per_token = 2
        cfg.moe_period           = 2
        cfg.moe_slot             = 0
        cfg.use_kan              = True
    else:
        cfg.num_layers = num_layers
        cfg.d_model = d_model
        cfg.n_heads = 8
        cfg.n_kv_heads = 4

        if mamba3:
            cfg.mamba3_enabled = True
            cfg.mamba3_state_dim = 128
        else:
            cfg.mamba3_enabled = False
            cfg.mamba2_faithful = True

        # All models are hybrid with Attention
        cfg.attention_period = 2
        cfg.attention_slot = 1

        cfg.use_kan = kan

        if moe:
            cfg.use_moe = True
            cfg.num_experts = 4
            cfg.num_experts_per_token = 2
            cfg.moe_period = 2
            cfg.moe_slot = 0
        else:
            cfg.use_moe = False

    model = nsos_ext.JambaModel(cfg, nsos_ext.Device.GPU)
    param_count = sum(p.data.size for p in model.parameters())
    return model, param_count



# -----------------------------------------------------------------------------
# 3. GPU Benchmark Runners
# -----------------------------------------------------------------------------
def run_gpu_flappy_trial(
    model_name: str,
    model: nsos_ext.JambaModel,
    lr: float,
    dataset: List[Tuple[List[int], List[int]]],
) -> Dict[str, float]:
    print(f"\n--- [GPU] Training {model_name} on Flappy Bird (LR = {lr}) ---", flush=True)
    trainer = nsos_ext.Trainer(model, lr)
    trainer.weight_decay = 0.0001
    trainer.max_grad_norm = 1.0

    t0 = time.time()
    losses = []
    initial_loss = None
    for i, (prompt, target) in enumerate(dataset, 1):
        loss = trainer.train_supervised(prompt, target)
        losses.append(loss)
        if initial_loss is None:
            initial_loss = loss
        if i in (1, len(dataset) // 2, len(dataset)):
            print(f"  [{model_name}] Step {i}/{len(dataset)} | Current Loss: {loss:.4f} | Running Avg: {np.mean(losses[-20:]):.4f}", flush=True)

    train_time = time.time() - t0
    final_loss = float(np.mean(losses[-15:]))
    steps_per_sec = len(dataset) / max(0.001, train_time)
    print(f"  [{model_name}] Completed in {train_time:.1f}s ({steps_per_sec:.1f} steps/s) | Initial: {initial_loss:.4f} -> Final: {final_loss:.4f}", flush=True)

    # Evaluate Flight
    model.set_training_mode(False)
    agent = NSOSAgent(model)
    test_env = FlappyEnvironment(seed=555)
    scores = []
    frames = []

    for ep in range(10):
        agent.reset_episode()
        obs = test_env.reset()
        done = False
        while not done and test_env.frames < 2000:
            act = agent.predict_action(obs)
            obs, _, done, sc = test_env.step(act)
        scores.append(sc)
        frames.append(test_env.frames)

    avg_score = float(np.mean(scores))
    max_score = float(max(scores))
    avg_frames = float(np.mean(frames))

    print(f"  [{model_name} Flight Trial] Avg Score: {avg_score:.1f} pipes | Max Score: {int(max_score)} | Avg Survival: {avg_frames:.1f} frames", flush=True)
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "avg_score": avg_score,
        "max_score": max_score,
        "avg_frames": avg_frames,
        "train_time": train_time,
        "steps_per_sec": steps_per_sec,
    }


def run_gpu_kv_trial(
    model_name: str,
    model: nsos_ext.JambaModel,
    lr: float,
    train_data: List[Tuple[List[int], List[int]]],
    test_data: List[Tuple[List[int], List[int]]],
) -> Dict[str, float]:
    print(f"\n--- [GPU] Task A: Associative Recall ({model_name}, LR = {lr}) ---", flush=True)

    def eval_acc(m, dataset):
        m.set_training_mode(False)
        correct = 0
        for seq, target in dataset:
            m.reset_session()
            logits = m.forward_ids(seq, None)
            pred = int(np.argmax(logits.numpy()[-1]))
            if pred == target[0]:
                correct += 1
        return 100.0 * correct / len(dataset)

    acc_before = eval_acc(model, test_data)
    model.set_training_mode(True)
    trainer = nsos_ext.Trainer(model, lr)
    trainer.max_grad_norm = 1.0

    t0 = time.time()
    losses = []
    for i, (p, t) in enumerate(train_data, 1):
        loss = trainer.train_supervised(p, t)
        losses.append(loss)
        if i in (1, len(train_data) // 2, len(train_data)):
            print(f"  [{model_name}] Step {i}/{len(train_data)} | Loss: {loss:.4f}", flush=True)

    train_time = time.time() - t0
    acc_after = eval_acc(model, test_data)
    print(f"  [{model_name}] Held-Out Accuracy: {acc_before:.1f}% -> {acc_after:.1f}% (in {train_time:.1f}s)", flush=True)
    return {"acc_before": acc_before, "acc_after": acc_after, "train_time": train_time, "final_loss": float(np.mean(losses[-10:]))}


def run_gpu_reversal_trial(
    model_name: str,
    model: nsos_ext.JambaModel,
    lr: float,
    train_data: List[Tuple[List[int], List[int]]],
    test_data: List[Tuple[List[int], List[int]]],
) -> Dict[str, float]:
    print(f"\n--- [GPU] Task B: Algorithmic Sequence Reversal ({model_name}, LR = {lr}) ---", flush=True)

    def eval_rev(m, dataset):
        m.set_training_mode(False)
        correct = 0
        for seq, target in dataset:
            m.reset_session()
            logits = m.forward_ids(seq, None)
            pred = int(np.argmax(logits.numpy()[-1]))
            if pred == target[0]:
                correct += 1
        return 100.0 * correct / len(dataset)

    acc_before = eval_rev(model, test_data)
    model.set_training_mode(True)
    trainer = nsos_ext.Trainer(model, lr)
    trainer.max_grad_norm = 1.0

    t0 = time.time()
    losses = []
    for i, (p, t) in enumerate(train_data, 1):
        loss = trainer.train_supervised(p, t)
        losses.append(loss)
        if i in (1, len(train_data) // 2, len(train_data)):
            print(f"  [{model_name}] Step {i}/{len(train_data)} | Loss: {loss:.4f}", flush=True)

    train_time = time.time() - t0
    acc_after = eval_rev(model, test_data)
    print(f"  [{model_name}] Held-Out Accuracy: {acc_before:.1f}% -> {acc_after:.1f}% (in {train_time:.1f}s)", flush=True)
    return {"acc_before": acc_before, "acc_after": acc_after, "train_time": train_time, "final_loss": float(np.mean(losses[-10:]))}


# -----------------------------------------------------------------------------
# 4. Main Suite
# -----------------------------------------------------------------------------
def main():
    print("=" * 88, flush=True)
    print("      NSOS DEDICATED GPU BENCHMARK: 4 HYBRID ARCHITECTURES ON RADEON RX 7600", flush=True)
    print("      Comparing Mamba-2 vs Mamba-3 vs BitFastKAN vs MoE (All with Attention)", flush=True)
    print("      Regime: 6.000 passos — benchmark definitivo de inteligencia", flush=True)
    print("=" * 88, flush=True)

    # Common Datasets — regime 6000 passos
    N_FLAPPY = 6000
    N_KV_TR  = 750
    N_KV_TE  = 100
    N_REV_TR = 600
    N_REV_TE = 100

    print(f"\n[Stage 0] Generating Common Datasets (Flappy={N_FLAPPY}, KV={N_KV_TR}/{N_KV_TE}, Rev={N_REV_TR}/{N_REV_TE})...", flush=True)

    # Flappy — multiplos episodios do expert ate atingir N_FLAPPY amostras
    flappy_env = FlappyEnvironment(seed=777)
    flappy_dataset: List[Tuple[List[int], List[int]]] = []
    while len(flappy_dataset) < N_FLAPPY:
        obs = flappy_env.reset()
        done = False
        while not done and len(flappy_dataset) < N_FLAPPY:
            act = flappy_expert_policy(obs)
            flappy_dataset.append((
                FlappyTokenizer.encode_state(obs),
                [FlappyTokenizer.ACTION_FLAP if act == 1 else FlappyTokenizer.ACTION_COAST],
            ))
            obs, _, done, _ = flappy_env.step(act)

    keys = list(range(10, 30))
    vals = list(range(35, 55))
    def make_kv(rng):
        sk = rng.sample(keys, 4)
        sv = [rng.choice(vals) for _ in range(4)]
        seq = []
        for k, v in zip(sk, sv):
            seq.extend([k, v])
        idx = rng.randint(0, 3)
        return seq + [sk[idx]], [sv[idx]]

    rng_kv = random.Random(42)
    kv_train = [make_kv(rng_kv) for _ in range(N_KV_TR)]
    kv_test  = [make_kv(rng_kv) for _ in range(N_KV_TE)]

    tokens_pool = list(range(10, 45))
    def make_rev(rng):
        seq = [rng.choice(tokens_pool) for _ in range(4)]
        return [1] + seq + [2], [seq[-1]]

    rng_rev = random.Random(88)
    rev_train = [make_rev(rng_rev) for _ in range(N_REV_TR)]
    rev_test  = [make_rev(rng_rev) for _ in range(N_REV_TE)]

    print(f"  -> Gerado: Flappy ({N_FLAPPY} passos expert), KV ({N_KV_TR} treino, {N_KV_TE} teste), Reversal ({N_REV_TR} treino, {N_REV_TE} teste)", flush=True)

    hybrid_defs = [
        ("Mamba-2 Híbrido (Attn)", dict(mamba3=False, kan=False, moe=False, all_layers=False, lr=0.015)),
        ("Mamba-3 Híbrido (Attn)", dict(mamba3=True,  kan=False, moe=False, all_layers=False, lr=0.002)),
        ("Mamba-3 + Attn + KAN",   dict(mamba3=True,  kan=True,  moe=False, all_layers=False, lr=0.002)),
        ("Mamba-3 + Attn + MoE",   dict(mamba3=True,  kan=False, moe=True,  all_layers=False, lr=0.002)),
        # Arquitetura completa: todas as camadas juntas simultaneamente
        ("Mamba-3 Full Stack",     dict(mamba3=True,  kan=True,  moe=True,  all_layers=True,  lr=0.002)),
    ]

    models_info = {}
    print("\n[Hybrid Architecture Specs (On Device: AMD Radeon RX 7600 GPU)]", flush=True)
    for name, params in hybrid_defs:
        m, p_count = build_gpu_model(name, mamba3=params["mamba3"], kan=params["kan"],
                                     moe=params["moe"], all_layers=params["all_layers"])
        models_info[name] = {"params": p_count, "lr": params["lr"], "cfg": params}
        print(f"  - {name:<26}: {p_count:>10,} params | LR = {params['lr']}", flush=True)

    results = {}

    for name, info in models_info.items():
        print(f"\n{'=' * 88}", flush=True)
        print(f"  EVALUATING ON GPU: {name.upper()}", flush=True)
        print(f"{'=' * 88}", flush=True)
        p = info["cfg"]
        lr = info["lr"]

        m_flappy, _ = build_gpu_model(name, mamba3=p["mamba3"], kan=p["kan"],
                                      moe=p["moe"], all_layers=p["all_layers"])
        res_flappy = run_gpu_flappy_trial(name, m_flappy, lr=lr, dataset=list(flappy_dataset))

        m_kv, _ = build_gpu_model(name, mamba3=p["mamba3"], kan=p["kan"],
                                  moe=p["moe"], all_layers=p["all_layers"])
        res_kv = run_gpu_kv_trial(name, m_kv, lr=lr, train_data=kv_train, test_data=kv_test)

        m_rev, _ = build_gpu_model(name, mamba3=p["mamba3"], kan=p["kan"],
                                   moe=p["moe"], all_layers=p["all_layers"])
        res_rev = run_gpu_reversal_trial(name, m_rev, lr=lr, train_data=rev_train, test_data=rev_test)

        results[name] = {
            "params": info["params"],
            "flappy": res_flappy,
            "kv": res_kv,
            "rev": res_rev,
        }

    # Final Scorecard Table
    print("\n" + "=" * 105, flush=True)
    print("                     FINAL GPU HYBRID SCORECARD (AMD RADEON RX 7600)", flush=True)
    print("=" * 105, flush=True)
    header = f"{'Architecture':<26} | {'Params':<10} | {'GPU Speed':<12} | {'Flappy (Avg)':<13} | {'Flappy (Max)':<12} | {'KV Recall':<10} | {'Reversal':<9}"
    print(header, flush=True)
    print("-" * 105, flush=True)
    for name, r in results.items():
        flappy_avg = f"{r['flappy']['avg_score']:.1f} pipes"
        flappy_max = f"{int(r['flappy']['max_score'])} pipes"
        gpu_speed = f"{r['flappy']['steps_per_sec']:.1f} st/s"
        kv_acc = f"{r['kv']['acc_after']:.1f}%"
        rev_acc = f"{r['rev']['acc_after']:.1f}%"
        params_str = f"{r['params']:,}"
        print(f"{name:<26} | {params_str:<10} | {gpu_speed:<12} | {flappy_avg:<13} | {flappy_max:<12} | {kv_acc:<10} | {rev_acc:<9}", flush=True)
    print("=" * 105, flush=True)


if __name__ == "__main__":
    t_start = time.time()
    main()
    print(f"\n[GPU BENCHMARK COMPLETE] Total elapsed: {time.time() - t_start:.1f}s", flush=True)
