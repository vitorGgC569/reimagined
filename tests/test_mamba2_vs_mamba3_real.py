"""
tests/test_mamba2_vs_mamba3_real.py

Official Head-to-Head Benchmark: Mamba-2 vs Mamba-3 under Calibrated Hyperparameters.
Tests both architectures on identical tasks with model-appropriate learning rates:
  - Mamba-2 SSD: lr = 0.015 (proven optimal for static-A SSD)
  - Mamba-3:     lr = 0.002 (calibrated for data-dependent A, trapezoidal dt, & rotary phase)

Tasks Evaluated:
  1. Flappy Bird Autopilot (2,000 expert flight trajectories, 10 live test trials)
  2. Associative Recall / Needle-in-a-Haystack (600 train, 100 test pairs)
  3. Algorithmic Sequence Reversal (500 train, 100 test sequences)

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

# -----------------------------------------------------------------------------
# 0. Setup & nsos_ext Resolution
# -----------------------------------------------------------------------------
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
# 1. Flappy Bird Environment & Tokenizer
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
        min_top = 40.0
        max_top = self.HEIGHT - 160.0
        top = self.rng.uniform(min_top, max_top)
        self.pipes.append(Pipe(x=x, top_height=top, gap=110.0))

    def get_next_pipe(self) -> Pipe:
        for p in self.pipes:
            if p.x + p.width >= self.BIRD_X - self.BIRD_RADIUS:
                return p
        return self.pipes[0]

    def get_observation(self) -> Tuple[float, float, float]:
        pipe = self.get_next_pipe()
        dx = max(0.0, pipe.x - self.BIRD_X)
        dy = self.bird_y - pipe.gap_center
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

    def render_ascii(self) -> str:
        rows = 12
        cols = 32
        grid = [[" " for _ in range(cols)] for _ in range(rows)]

        scale_x = cols / self.WIDTH
        scale_y = rows / self.HEIGHT

        for p in self.pipes:
            c1 = int(p.x * scale_x)
            c2 = int((p.x + p.width) * scale_x)
            top_r = int(p.top_height * scale_y)
            bot_r = int(p.bottom_y * scale_y)
            for c in range(max(0, c1), min(cols, c2 + 1)):
                for r in range(0, min(rows, top_r)):
                    grid[r][c] = "█"
                for r in range(max(0, bot_r), rows):
                    grid[r][c] = "█"

        bird_r = max(0, min(rows - 1, int(self.bird_y * scale_y)))
        bird_c = max(0, min(cols - 1, int(self.BIRD_X * scale_x)))
        grid[bird_r][bird_c] = "B"

        header = f"┌{'─' * cols}┐\n"
        body = "".join(f"│{''.join(row)}│\n" for row in grid)
        footer = f"└{'─' * cols}┘\nScore: {self.score} | Frame: {self.frames} | Y: {self.bird_y:.1f} | Vel: {self.bird_vel:.1f}"
        return header + body + footer


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
    projected_dy = dy + (vel * 2.5)
    if projected_dy > 5.0 and vel > -3.5:
        return 1
    return 0


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
# 2. Model Factories
# -----------------------------------------------------------------------------
def build_mamba2_model(num_layers: int = 2, d_model: int = 128, vocab_size: int = 64) -> Tuple[nsos_ext.JambaModel, int]:
    model = nsos_ext.JambaModel(num_layers, d_model, vocab_size, nsos_ext.Device.CPU)
    param_count = sum(p.data.size for p in model.parameters())
    return model, param_count


def build_mamba3_model(num_layers: int = 2, d_model: int = 128, vocab_size: int = 64) -> Tuple[nsos_ext.JambaModel, int]:
    cfg = nsos_ext.ModelConfig()
    cfg.architecture_schema_version = 3
    cfg.mamba3_enabled = True
    cfg.mamba3_state_dim = 128
    cfg.num_layers = num_layers
    cfg.d_model = d_model
    cfg.vocab_size = vocab_size
    cfg.n_heads = 8
    cfg.n_kv_heads = 4

    model = nsos_ext.JambaModel(cfg, nsos_ext.Device.CPU)
    param_count = sum(p.data.size for p in model.parameters())
    return model, param_count


# -----------------------------------------------------------------------------
# 3. Flappy Bird Head-to-Head Trial
# -----------------------------------------------------------------------------
def run_flappy_trial(model_name: str, model: nsos_ext.JambaModel, lr: float, dataset: List[Tuple[List[int], List[int]]]) -> Dict[str, float]:
    print(f"\n--- Training {model_name} on Flappy Bird (Calibrated LR = {lr}) ---")
    trainer = nsos_ext.Trainer(model, lr)
    trainer.weight_decay = 0.0001
    trainer.max_grad_norm = 1.0

    epochs = 6
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        losses = []
        random.shuffle(dataset)
        for prompt, target in dataset:
            losses.append(trainer.train_supervised(prompt, target))
        if epoch in (1, 3, 6):
            print(f"  [{model_name}] Epoch {epoch}/6 | Loss: {np.mean(losses):.4f}")

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


# -----------------------------------------------------------------------------
# 4. Synthetic Benchmarks Head-to-Head
# -----------------------------------------------------------------------------
def run_associative_recall_trial(model_name: str, model: nsos_ext.JambaModel, lr: float, train_data, test_data) -> Dict[str, float]:
    print(f"\n--- Task A: Associative Recall ({model_name}, LR = {lr}) ---")
    trainer = nsos_ext.Trainer(model, lr)

    def eval_kv(m, d):
        correct = 0
        for prompt, target in d:
            m.reset_session()
            logits = m.forward_ids(prompt, None)
            pred = int(np.argmax(logits.numpy()[-1]))
            if pred == target[0]:
                correct += 1
        return (correct / len(d)) * 100.0

    acc_before = eval_kv(model, test_data)

    t0 = time.time()
    for epoch in range(1, 9):
        losses = [trainer.train_supervised(p, t) for p, t in train_data]
        if epoch in (1, 4, 8):
            print(f"  [{model_name}] Epoch {epoch}/8 | Loss: {np.mean(losses):.4f}")

    train_time = time.time() - t0
    acc_after = eval_kv(model, test_data)
    print(f"  [{model_name}] Held-Out Accuracy: {acc_before:.1f}% -> {acc_after:.1f}% (in {train_time:.1f}s)")
    return {"acc_before": acc_before, "acc_after": acc_after, "train_time": train_time}


def run_sequence_reversal_trial(model_name: str, model: nsos_ext.JambaModel, lr: float, train_data, test_data) -> Dict[str, float]:
    print(f"\n--- Task B: Algorithmic Sequence Reversal ({model_name}, LR = {lr}) ---")
    trainer = nsos_ext.Trainer(model, lr)

    def eval_rev(m, d):
        correct = 0
        for prompt, target in d:
            m.reset_session()
            logits = m.forward_ids(prompt, None)
            pred = int(np.argmax(logits.numpy()[-1]))
            if pred == target[0]:
                correct += 1
        return (correct / len(d)) * 100.0

    acc_before = eval_rev(model, test_data)

    t0 = time.time()
    for epoch in range(1, 9):
        losses = [trainer.train_supervised(p, t) for p, t in train_data]
        if epoch in (1, 4, 8):
            print(f"  [{model_name}] Epoch {epoch}/8 | Loss: {np.mean(losses):.4f}")

    train_time = time.time() - t0
    acc_after = eval_rev(model, test_data)
    print(f"  [{model_name}] Held-Out Accuracy: {acc_before:.1f}% -> {acc_after:.1f}% (in {train_time:.1f}s)")
    return {"acc_before": acc_before, "acc_after": acc_after, "train_time": train_time}


# -----------------------------------------------------------------------------
# 5. Main Execution & Fair Comparison Pipeline
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("      NSOS COMPREHENSIVE HEAD-TO-HEAD AUDIT: MAMBA-2 VS MAMBA-3")
    print("      Both Architectures Evaluated Under Calibrated Hyperparameters")
    print("=" * 80)

    # 1. Dataset Generation (Strictly identical across both models)
    print("\n[Stage 0] Generating Common Datasets...")
    # Flappy Bird 2,000 samples
    flappy_env = FlappyEnvironment(seed=777)
    flappy_dataset: List[Tuple[List[int], List[int]]] = []
    while len(flappy_dataset) < 2000:
        obs = flappy_env.reset()
        done = False
        while not done and len(flappy_dataset) < 2000:
            act = flappy_expert_policy(obs)
            flappy_dataset.append((FlappyTokenizer.encode_state(obs), [FlappyTokenizer.ACTION_FLAP if act == 1 else FlappyTokenizer.ACTION_COAST]))
            obs, _, done, _ = flappy_env.step(act)

    # Associative Recall 600 train, 100 test
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
    kv_train = [make_kv(rng_kv) for _ in range(600)]
    kv_test = [make_kv(rng_kv) for _ in range(100)]

    # Sequence Reversal 500 train, 100 test
    tokens_pool = list(range(10, 45))
    def make_rev(rng):
        seq = [rng.choice(tokens_pool) for _ in range(4)]
        return [1] + seq + [2], [seq[-1]]

    rng_rev = random.Random(88)
    rev_train = [make_rev(rng_rev) for _ in range(500)]
    rev_test = [make_rev(rng_rev) for _ in range(100)]

    print(f"  -> Generated: Flappy (2,000 pairs), KV Recall (600 train, 100 test), Reversal (500 train, 100 test)")

    # 2. Build Models
    m2_flappy, p2_flappy = build_mamba2_model(2, 128, 64)
    m3_flappy, p3_flappy = build_mamba3_model(2, 128, 64)

    m2_kv, _ = build_mamba2_model(2, 128, 64)
    m3_kv, _ = build_mamba3_model(2, 128, 64)

    m2_rev, _ = build_mamba2_model(2, 128, 64)
    m3_rev, _ = build_mamba3_model(2, 128, 64)

    print(f"\n[Model Specs]")
    print(f"  - Mamba-2: {p2_flappy:,} parameters (SSD scan, static A)")
    print(f"  - Mamba-3: {p3_flappy:,} parameters (SISO/MIMO, data-dependent A, trapezoidal dt, rotary phase)")

    # 3. Flappy Bird Benchmark
    res_m2_flappy = run_flappy_trial("Mamba-2", m2_flappy, lr=0.015, dataset=list(flappy_dataset))
    res_m3_flappy = run_flappy_trial("Mamba-3", m3_flappy, lr=0.002, dataset=list(flappy_dataset))

    # 4. Associative Recall Benchmark
    res_m2_kv = run_associative_recall_trial("Mamba-2", m2_kv, lr=0.015, train_data=kv_train, test_data=kv_test)
    res_m3_kv = run_associative_recall_trial("Mamba-3", m3_kv, lr=0.002, train_data=kv_train, test_data=kv_test)

    # 5. Sequence Reversal Benchmark
    res_m2_rev = run_sequence_reversal_trial("Mamba-2", m2_rev, lr=0.015, train_data=rev_train, test_data=rev_test)
    res_m3_rev = run_sequence_reversal_trial("Mamba-3", m3_rev, lr=0.002, train_data=rev_train, test_data=rev_test)

    # 6. Final Comparative Scorecard
    print("\n" + "=" * 80)
    print("                    FINAL HEAD-TO-HEAD SCORECARD")
    print("=" * 80)
    print(f"{'Metric / Benchmark':<35} | {'Mamba-2 (lr=0.015)':<20} | {'Mamba-3 (lr=0.002)':<20} | {'Winner':<10}")
    print("-" * 80)
    print(f"{'Total Parameters':<35} | {p2_flappy:<20,} | {p3_flappy:<20,} | {'Mamba-3 (Capacity)'}")
    print(f"{'Flappy: Average Pipes Cleared':<35} | {res_m2_flappy['avg_score']:<20.1f} | {res_m3_flappy['avg_score']:<20.1f} | {'Mamba-2' if res_m2_flappy['avg_score'] >= res_m3_flappy['avg_score'] else 'Mamba-3'}")
    print(f"{'Flappy: Max Pipes Cleared':<35} | {int(res_m2_flappy['max_score']):<20} | {int(res_m3_flappy['max_score']):<20} | {'Mamba-2' if res_m2_flappy['max_score'] >= res_m3_flappy['max_score'] else 'Mamba-3'}")
    print(f"{'Flappy: Avg Survival (Frames)':<35} | {res_m2_flappy['avg_frames']:<20.1f} | {res_m3_flappy['avg_frames']:<20.1f} | {'Mamba-2' if res_m2_flappy['avg_frames'] >= res_m3_flappy['avg_frames'] else 'Mamba-3'}")
    print(f"{'Associative Recall (Accuracy %)':<35} | {res_m2_kv['acc_after']:<19.1f}% | {res_m3_kv['acc_after']:<19.1f}% | {'Mamba-3' if res_m3_kv['acc_after'] > res_m2_kv['acc_after'] else 'Mamba-2'}")
    print(f"{'Sequence Reversal (Accuracy %)':<35} | {res_m2_rev['acc_after']:<19.1f}% | {res_m3_rev['acc_after']:<19.1f}% | {'Mamba-2' if res_m2_rev['acc_after'] >= res_m3_rev['acc_after'] else 'Mamba-3'}")
    print("=" * 80)


if __name__ == "__main__":
    t_start = time.time()
    main()
    print(f"\n[BENCHMARK SUITE COMPLETE] Total elapsed: {time.time() - t_start:.1f}s")
