"""
tests/test_hybrid_stack_benchmark.py

Comprehensive NSOS Architectural Benchmark:
Evaluating the Full Hybrid Layer Stack (Mamba-2 vs Mamba-3 vs Jamba Attention vs BitFastKAN).

Architectures Evaluated (all 4 layers, d_model=128, vocab_size=64):
  1. Mamba-2 Puro: Pure Mamba-2 SSD recurrence + dense MLP FFN (lr=0.015)
  2. Mamba-3 Puro: Pure Mamba-3 continuous recurrence + dense MLP FFN (lr=0.002)
  3. Mamba-3 + Atenção GQA: Jamba Hybrid (interleaving Mamba-3 and GQA/RoPE attention) + dense MLP FFN (lr=0.002)
  4. Mamba-3 + Atenção + KAN: Omni Hybrid (Mamba-3 + GQA attention + BitFastKAN RBF splines) (lr=0.002)

Tasks Evaluated:
  1. Flappy Bird Autopilot (1,000 expert trajectories, 10 live flight trials)
  2. Associative Recall / Needle-in-a-Haystack (400 train, 100 test pairs)
  3. Algorithmic Sequence Reversal (300 train, 100 test sequences)

Strict Rules:
  - 100% Native C++ NSOS JambaModel & Trainer runtime via nsos_ext.
  - Zero mocks, stubs, or simplifications.
  - Zero modifications outside the tests/ directory.
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

CPU_BUILD_DIR = REPO_ROOT / "OXN" / "nsos" / "build-gm-cpu"
if not CPU_BUILD_DIR.exists() or not list(CPU_BUILD_DIR.glob("nsos_ext*.pyd")):
    raise RuntimeError(f"Could not find compiled nsos_ext in {CPU_BUILD_DIR}")

sys.path.insert(0, str(CPU_BUILD_DIR))
import nsos_ext
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
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

    def predict_action(self, obs: Tuple[float, float, float]) -> int:
        self.model.reset_session()
        prompt = FlappyTokenizer.encode_state(obs)
        logits_tensor = self.model.forward_ids(prompt, None)
        logits_np = logits_tensor.numpy()[-1]

        coast_score = float(logits_np[FlappyTokenizer.ACTION_COAST])
        flap_score = float(logits_np[FlappyTokenizer.ACTION_FLAP])

        return 1 if flap_score > coast_score else 0


# -----------------------------------------------------------------------------
# 2. Model Factory for Hybrid Layers
# -----------------------------------------------------------------------------
def build_model(
    name: str,
    mamba3: bool,
    attention: bool,
    kan: bool,
    moe: bool = False,
    num_layers: int = 4,
    d_model: int = 128,
    vocab_size: int = 64,
) -> Tuple[nsos_ext.JambaModel, int]:
    cfg = nsos_ext.ModelConfig()
    cfg.architecture_schema_version = 3
    cfg.num_layers = num_layers
    cfg.d_model = d_model
    cfg.vocab_size = vocab_size
    cfg.n_heads = 8
    cfg.n_kv_heads = 4

    if mamba3:
        cfg.mamba3_enabled = True
        cfg.mamba3_state_dim = 128
    else:
        cfg.mamba3_enabled = False
        cfg.mamba2_faithful = True

    if attention:
        cfg.attention_period = 2
        cfg.attention_slot = 1
    else:
        cfg.attention_period = 999
        cfg.attention_slot = 998

    cfg.use_kan = kan

    if moe:
        cfg.use_moe = True
        cfg.num_experts = 4
        cfg.num_experts_per_token = 2
        cfg.moe_period = 2
        cfg.moe_slot = 0
    else:
        cfg.use_moe = False

    model = nsos_ext.JambaModel(cfg, nsos_ext.Device.CPU)
    param_count = sum(p.data.size for p in model.parameters())
    return model, param_count


# -----------------------------------------------------------------------------
# 3. Benchmark Runners
# -----------------------------------------------------------------------------
def run_flappy_trial(
    model_name: str,
    model: nsos_ext.JambaModel,
    lr: float,
    dataset: List[Tuple[List[int], List[int]]],
    epochs: int = 4,
) -> Dict[str, float]:
    print(f"\n--- Training {model_name} on Flappy Bird (LR = {lr}) ---")
    trainer = nsos_ext.Trainer(model, lr)
    trainer.weight_decay = 0.0001
    trainer.max_grad_norm = 1.0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        losses = []
        random.shuffle(dataset)
        for prompt, target in dataset:
            losses.append(trainer.train_supervised(prompt, target))
        if epoch in (1, 2, epochs):
            print(f"  [{model_name}] Epoch {epoch}/{epochs} | Loss: {np.mean(losses):.4f}")

    train_time = time.time() - t0
    final_loss = float(np.mean(losses))
    print(f"  [{model_name}] Completed in {train_time:.1f}s | Final Loss: {final_loss:.4f}")

    # Evaluate Flight
    agent = NSOSAgent(model)
    test_env = FlappyEnvironment(seed=555)
    scores = []
    frames = []

    for ep in range(10):
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

    print(f"  [{model_name} Flight Trial] Avg Score: {avg_score:.1f} pipes | Max Score: {int(max_score)} | Avg Survival: {avg_frames:.1f} frames")
    return {
        "final_loss": final_loss,
        "avg_score": avg_score,
        "max_score": max_score,
        "avg_frames": avg_frames,
        "train_time": train_time,
    }


def run_associative_recall_trial(
    model_name: str,
    model: nsos_ext.JambaModel,
    lr: float,
    train_data: List[Tuple[List[int], List[int]]],
    test_data: List[Tuple[List[int], List[int]]],
    epochs: int = 6,
) -> Dict[str, float]:
    print(f"\n--- Task A: Associative Recall ({model_name}, LR = {lr}) ---")

    def eval_acc(m, dataset):
        correct = 0
        for seq, target in dataset:
            m.reset_session()
            logits = m.forward_ids(seq, None)
            pred = int(np.argmax(logits.numpy()[-1]))
            if pred == target[0]:
                correct += 1
        return 100.0 * correct / len(dataset)

    acc_before = eval_acc(model, test_data)
    trainer = nsos_ext.Trainer(model, lr)
    trainer.max_grad_norm = 1.0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        losses = [trainer.train_supervised(p, t) for p, t in train_data]
        if epoch in (1, 3, epochs):
            print(f"  [{model_name}] Epoch {epoch}/{epochs} | Loss: {np.mean(losses):.4f}")

    train_time = time.time() - t0
    acc_after = eval_acc(model, test_data)
    print(f"  [{model_name}] Held-Out Accuracy: {acc_before:.1f}% -> {acc_after:.1f}% (in {train_time:.1f}s)")
    return {"acc_before": acc_before, "acc_after": acc_after, "train_time": train_time}


def run_sequence_reversal_trial(
    model_name: str,
    model: nsos_ext.JambaModel,
    lr: float,
    train_data: List[Tuple[List[int], List[int]]],
    test_data: List[Tuple[List[int], List[int]]],
    epochs: int = 6,
) -> Dict[str, float]:
    print(f"\n--- Task B: Algorithmic Sequence Reversal ({model_name}, LR = {lr}) ---")

    def eval_rev(m, dataset):
        correct = 0
        for seq, target in dataset:
            m.reset_session()
            logits = m.forward_ids(seq, None)
            pred = int(np.argmax(logits.numpy()[-1]))
            if pred == target[0]:
                correct += 1
        return 100.0 * correct / len(dataset)

    acc_before = eval_rev(model, test_data)
    trainer = nsos_ext.Trainer(model, lr)
    trainer.max_grad_norm = 1.0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        losses = [trainer.train_supervised(p, t) for p, t in train_data]
        if epoch in (1, 3, epochs):
            print(f"  [{model_name}] Epoch {epoch}/{epochs} | Loss: {np.mean(losses):.4f}")

    train_time = time.time() - t0
    acc_after = eval_rev(model, test_data)
    print(f"  [{model_name}] Held-Out Accuracy: {acc_before:.1f}% -> {acc_after:.1f}% (in {train_time:.1f}s)")
    return {"acc_before": acc_before, "acc_after": acc_after, "train_time": train_time}


# -----------------------------------------------------------------------------
# 4. Main Suite Execution
# -----------------------------------------------------------------------------
def main():
    print("=" * 85)
    print("       NSOS ARCHITECTURAL BENCHMARK: FULL HYBRID LAYER STACK")
    print("  Comparing Mamba-2 vs Mamba-3 vs Jamba Attention vs BitFastKAN")
    print("=" * 85)

    # 1. Generate Common Datasets
    print("\n[Stage 0] Generating Common Datasets...")
    # Flappy Bird 1,000 samples
    flappy_env = FlappyEnvironment(seed=777)
    flappy_dataset: List[Tuple[List[int], List[int]]] = []
    while len(flappy_dataset) < 1000:
        obs = flappy_env.reset()
        done = False
        while not done and len(flappy_dataset) < 1000:
            act = flappy_expert_policy(obs)
            flappy_dataset.append((FlappyTokenizer.encode_state(obs), [FlappyTokenizer.ACTION_FLAP if act == 1 else FlappyTokenizer.ACTION_COAST]))
            obs, _, done, _ = flappy_env.step(act)

    # Associative Recall 400 train, 100 test
    keys = list(range(10, 30))
    vals = list(range(35, 55))
    def make_kv(rng):
        sk = rng.sample(keys, 4)
        sv = [rng.choice(vals) for _ in range(4)]
        seq = []
        for k, v in zip(sk, sv): seq.extend([k, v])
        idx = rng.randint(0, 3)
        return seq + [sk[idx]], [sv[idx]]

    rng_kv = random.Random(42)
    kv_train = [make_kv(rng_kv) for _ in range(400)]
    kv_test = [make_kv(rng_kv) for _ in range(100)]

    # Sequence Reversal 300 train, 100 test
    tokens_pool = list(range(10, 45))
    def make_rev(rng):
        seq = [rng.choice(tokens_pool) for _ in range(4)]
        return [1] + seq + [2], [seq[-1]]

    rng_rev = random.Random(88)
    rev_train = [make_rev(rng_rev) for _ in range(300)]
    rev_test = [make_rev(rng_rev) for _ in range(100)]

    print(f"  -> Generated: Flappy (1,000 pairs), KV Recall (400 train, 100 test), Reversal (300 train, 100 test)")

    # Model specifications to test
    model_defs = [
        ("Mamba-2 Puro", dict(mamba3=False, attention=False, kan=False, lr=0.015)),
        ("Mamba-3 Puro", dict(mamba3=True, attention=False, kan=False, lr=0.002)),
        ("Mamba-3 + Atenção GQA", dict(mamba3=True, attention=True, kan=False, lr=0.002)),
        ("Mamba-3 + Atenção + KAN", dict(mamba3=True, attention=True, kan=True, lr=0.002)),
    ]

    models_info = {}
    print("\n[Model Specifications (All 4 layers, d_model=128)]")
    for name, params in model_defs:
        m, p_count = build_model(
            name,
            mamba3=params["mamba3"],
            attention=params["attention"],
            kan=params["kan"],
        )
        models_info[name] = {"params": p_count, "lr": params["lr"], "cfg": params}
        print(f"  - {name:<26}: {p_count:>9,} parameters | Base LR = {params['lr']}")

    results = {}

    for name, info in models_info.items():
        print(f"\n{'=' * 85}")
        print(f"  EVALUATING ARCHITECTURE: {name.upper()}")
        print(f"{'=' * 85}")
        p = info["cfg"]
        lr = info["lr"]

        # Instantiate fresh model for Flappy
        m_flappy, _ = build_model(name, mamba3=p["mamba3"], attention=p["attention"], kan=p["kan"])
        res_flappy = run_flappy_trial(name, m_flappy, lr=lr, dataset=list(flappy_dataset), epochs=4)

        # Instantiate fresh model for KV Recall
        m_kv, _ = build_model(name, mamba3=p["mamba3"], attention=p["attention"], kan=p["kan"])
        res_kv = run_associative_recall_trial(name, m_kv, lr=lr, train_data=kv_train, test_data=kv_test, epochs=6)

        # Instantiate fresh model for Reversal
        m_rev, _ = build_model(name, mamba3=p["mamba3"], attention=p["attention"], kan=p["kan"])
        res_rev = run_sequence_reversal_trial(name, m_rev, lr=lr, train_data=rev_train, test_data=rev_test, epochs=6)

        results[name] = {
            "params": info["params"],
            "flappy": res_flappy,
            "kv": res_kv,
            "rev": res_rev,
        }

    # Final Scorecard Table
    print("\n" + "=" * 95)
    print("                     FINAL COMPREHENSIVE ARCHITECTURAL SCORECARD")
    print("=" * 95)
    header = f"{'Architecture':<26} | {'Params':<10} | {'Flappy (Avg)':<12} | {'Flappy (Max)':<12} | {'KV Recall':<12} | {'Reversal':<10}"
    print(header)
    print("-" * 95)
    for name, r in results.items():
        flappy_avg = f"{r['flappy']['avg_score']:.1f} pipes"
        flappy_max = f"{int(r['flappy']['max_score'])} pipes"
        kv_acc = f"{r['kv']['acc_after']:.1f}%"
        rev_acc = f"{r['rev']['acc_after']:.1f}%"
        params_str = f"{r['params']:,}"
        print(f"{name:<26} | {params_str:<10} | {flappy_avg:<12} | {flappy_max:<12} | {kv_acc:<12} | {rev_acc:<10}")
    print("=" * 95)


if __name__ == "__main__":
    t_start = time.time()
    main()
    print(f"\n[BENCHMARK SUITE COMPLETE] Total elapsed: {time.time() - t_start:.1f}s")
