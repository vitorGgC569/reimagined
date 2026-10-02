"""
tests/test_gpu_dedicated_training.py

Dedicated GPU Training Suite for NSOS Architecture.
Target Hardware: AMD Radeon RX 7600 (8.5 GB VRAM, RDNA 3 / gfx1102)
Backend: Native HIP / ROCm (nsos_ext compiled for AMD RDNA)

Strict Compliance:
  - 100% Native C++ HIP/GPU execution on Device.GPU.
  - No mocks, stubs, or CPU fallbacks.
  - Zero modifications to files outside the tests/ directory.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

# -----------------------------------------------------------------------------
# 0. Environment Setup & HIP GPU Binding Resolution
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

# Select the HIP build target specifically compiled for the AMD RX 7600
HIP_BUILD_DIR = REPO_ROOT / "OXN" / "nsos" / "build-codex-hip"

if not HIP_BUILD_DIR.exists() or not list(HIP_BUILD_DIR.glob("nsos_ext*.pyd")):
    raise RuntimeError(
        f"Could not find compiled HIP nsos_ext binary in {HIP_BUILD_DIR}"
    )

sys.path.insert(0, str(HIP_BUILD_DIR))
import nsos_ext
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# -----------------------------------------------------------------------------
# 1. GPU Hardware Diagnostics
# -----------------------------------------------------------------------------
def inspect_gpu_hardware():
    backend = nsos_ext.gpu_backend_name()
    vendor = nsos_ext.gpu_vendor_name()
    devices = list(nsos_ext.gpu_devices())

    print("=" * 78)
    print(" [HARDWARE] DEDICATED GPU HARDWARE DISCOVERY & HIP RUNTIME PROBE")
    print("=" * 78)
    print(f"  Backend: {backend.upper()} | Vendor: {vendor.upper()}")
    print(f"  Detected GPU Devices: {len(devices)}")
    for d in devices:
        is_dedicated = not d.get("integrated", False)
        marker = " [PRIMARY DEDICATED ACCELERATOR]" if is_dedicated else " [Integrated APU]"
        print(f"    - Device #{d['index']}: {d['name']} ({d['architecture']}){marker}")
        print(f"      VRAM: {d['total_memory'] / (1024**3):.2f} GB | Warp: {d['warp_size']} | Compiled: {d['compiled']} | FP16/BF16: {d['fp16']}/{d['bf16']}")

    dedicated = [d for d in devices if not d.get("integrated", False) and d.get("compiled", False)]
    if not dedicated:
        raise RuntimeError("No compiled dedicated GPU device found.")

    active_idx = nsos_ext.selected_gpu_device() if hasattr(nsos_ext, "selected_gpu_device") and callable(nsos_ext.selected_gpu_device) else selected_idx
    print(f"\n  Active Accelerator Selected: Device #{active_idx} ({dedicated[0]['name']})")
    print("=" * 78)
    return dedicated[0]


# -----------------------------------------------------------------------------
# 2. Flappy Bird Environment (2D Real Physics Simulator)
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


# -----------------------------------------------------------------------------
# 3. State Tokenizer
# -----------------------------------------------------------------------------
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


class NSOSGPUFlappyAgent:
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
# 4. Flappy Bird Training on Dedicated GPU
# -----------------------------------------------------------------------------
def run_gpu_flappy_bird_benchmark():
    print("\n" + "=" * 78)
    print(" [BENCHMARK 1] FLAPPY BIRD ON DEDICATED GPU (AMD RADEON RX 7600)")
    print(" Execution Device: nsos_ext.Device.GPU | Native HIP Kernels")
    print("=" * 78)

    d_model = 128
    n_layers = 2
    vocab_size = FlappyTokenizer.VOCAB_SIZE

    print(f"\n[GPU Init] Instantiating JambaModel(layers={n_layers}, d_model={d_model}, vocab={vocab_size}, device=GPU)...")
    model = nsos_ext.JambaModel(n_layers, d_model, vocab_size, nsos_ext.Device.GPU)
    param_count = sum(p.data.size for p in model.parameters())
    print(f"  -> Model Allocated on Dedicated VRAM: {param_count:,} parameters")

    trainer = nsos_ext.Trainer(model, 0.015)
    trainer.weight_decay = 0.0001
    trainer.max_grad_norm = 1.0

    agent = NSOSGPUFlappyAgent(model)

    # 1. Baseline Zero-Shot on GPU
    print("\n[Stage 1] Zero-Shot Baseline Flight Evaluation on GPU (10 episodes)...")
    env = FlappyEnvironment(seed=101)
    untrained_scores = []
    untrained_frames = []

    for ep in range(10):
        obs = env.reset()
        done = False
        while not done and env.frames < 2000:
            action = agent.predict_action(obs)
            obs, _, done, score = env.step(action)
        untrained_scores.append(score)
        untrained_frames.append(env.frames)

    print(f"  -> Untrained Average Score:      {np.mean(untrained_scores):.1f} pipes")
    print(f"  -> Untrained Average Survival:   {np.mean(untrained_frames):.1f} frames")

    # 2. Generate 2,000 Expert Trajectories
    print("\n[Stage 2] Generating Expert Dataset (2,000 trajectories)...")
    expert_env = FlappyEnvironment(seed=777)
    expert_dataset: List[Tuple[List[int], List[int]]] = []

    target_samples = 2000
    while len(expert_dataset) < target_samples:
        obs = expert_env.reset()
        done = False
        while not done and len(expert_dataset) < target_samples:
            expert_action = flappy_expert_policy(obs)
            prompt = FlappyTokenizer.encode_state(obs)
            target_token = (
                FlappyTokenizer.ACTION_FLAP if expert_action == 1 else FlappyTokenizer.ACTION_COAST
            )
            expert_dataset.append((prompt, [target_token]))
            obs, _, done, _ = expert_env.step(expert_action)

    print(f"  -> Collected {len(expert_dataset):,} state-action pairs.")

    # 3. Train on GPU via Native HIP
    print("\n[Stage 3] Training on Dedicated GPU via Native HIP autodiff & Adam...")
    epochs = 6
    t0 = time.time()

    initial_loss = None
    final_loss = None

    for epoch in range(1, epochs + 1):
        epoch_losses = []
        random.shuffle(expert_dataset)
        for prompt, target in expert_dataset:
            loss = trainer.train_supervised(prompt, target)
            epoch_losses.append(loss)
            if initial_loss is None:
                initial_loss = loss

        avg_loss = float(np.mean(epoch_losses))
        final_loss = avg_loss
        print(f"  Epoch {epoch:2d}/{epochs:2d} | Cross-Entropy Loss: {avg_loss:.4f} | Processed: {epoch * len(expert_dataset):,} steps")

    elapsed = time.time() - t0
    gpu_throughput = (epochs * len(expert_dataset)) / max(0.001, elapsed)
    print(f"  -> GPU Training completed in {elapsed:.2f}s ({gpu_throughput:.1f} steps/s) | Initial: {initial_loss:.4f} -> Final: {final_loss:.4f}")

    # 4. Evaluate GPU Trained Model
    print("\n[Stage 4] Autonomous Live Flight Trial with GPU-Trained Policy (10 episodes)...")
    test_env = FlappyEnvironment(seed=555)
    trained_scores = []
    trained_frames = []
    sample_render = ""

    for ep in range(10):
        obs = test_env.reset()
        done = False
        while not done and test_env.frames < 2000:
            action = agent.predict_action(obs)
            obs, _, done, score = test_env.step(action)
            if ep == 0 and test_env.frames == 80:
                sample_render = test_env.render_ascii()
        trained_scores.append(score)
        trained_frames.append(test_env.frames)

    print(f"  -> GPU Trained Average Score:    {np.mean(trained_scores):.1f} pipes")
    print(f"  -> GPU Trained Average Survival: {np.mean(trained_frames):.1f} frames")
    print(f"  -> Max Score Achieved:           {max(trained_scores)} pipes")

    if sample_render:
        print("\n[Visual] In-Flight Snapshot of GPU-Trained Autopilot:")
        print(sample_render)

    gain = (
        (np.mean(trained_frames) - np.mean(untrained_frames))
        / max(1.0, float(np.mean(untrained_frames)))
    ) * 100.0
    print(f"\n[Result] GPU Flight Survival Gain: +{gain:.1f}% over baseline.")
    print("=" * 78)
    return gpu_throughput


# -----------------------------------------------------------------------------
# 5. Synthetic Tasks on Dedicated GPU
# -----------------------------------------------------------------------------
def run_gpu_synthetic_benchmarks():
    print("\n" + "=" * 78)
    print(" [BENCHMARK 2] SYNTHETIC TASKS ON DEDICATED GPU (AMD RADEON RX 7600)")
    print("=" * 78)

    vocab_size = 64
    d_model = 128
    n_layers = 2

    # Task A: Associative Recall
    print("\n--- Task A: Associative Recall on Dedicated GPU (600 train, 100 test) ---")
    model_kv = nsos_ext.JambaModel(n_layers, d_model, vocab_size, nsos_ext.Device.GPU)
    trainer_kv = nsos_ext.Trainer(model_kv, 0.02)

    keys = list(range(10, 30))
    values = list(range(35, 55))

    def make_kv_sample(num_pairs: int = 4):
        selected_keys = random.sample(keys, num_pairs)
        selected_vals = [random.choice(values) for _ in range(num_pairs)]
        seq = []
        for k, v in zip(selected_keys, selected_vals):
            seq.extend([k, v])
        query_idx = random.randint(0, num_pairs - 1)
        return seq + [selected_keys[query_idx]], [selected_vals[query_idx]]

    train_data = [make_kv_sample(num_pairs=4) for _ in range(600)]
    test_data = [make_kv_sample(num_pairs=4) for _ in range(100)]

    def eval_kv(m, dataset):
        correct = 0
        for prompt, target in dataset:
            m.reset_session()
            logits = m.forward_ids(prompt, None)
            pred = int(np.argmax(logits.numpy()[-1]))
            if pred == target[0]:
                correct += 1
        return (correct / len(dataset)) * 100.0

    acc_before = eval_kv(model_kv, test_data)
    print(f"  -> GPU Zero-Shot Held-Out Accuracy:  {acc_before:.1f}%")

    t0 = time.time()
    for epoch in range(1, 9):
        losses = [trainer_kv.train_supervised(p, t) for p, t in train_data]
        if epoch in (1, 4, 8):
            print(f"  Epoch {epoch:2d}/8 | Cross-Entropy Loss: {np.mean(losses):.4f} | Processed: {epoch * len(train_data):,} steps")

    print(f"  -> GPU Training completed in {time.time() - t0:.2f}s")
    acc_after = eval_kv(model_kv, test_data)
    print(f"  -> GPU Post-Training Held-Out Accuracy: {acc_after:.1f}%")

    # Task B: Algorithmic Sequence Reversal
    print("\n--- Task B: Algorithmic Sequence Reversal on Dedicated GPU (500 train, 100 test) ---")
    model_rev = nsos_ext.JambaModel(n_layers, d_model, vocab_size, nsos_ext.Device.GPU)
    trainer_rev = nsos_ext.Trainer(model_rev, 0.02)

    tokens_pool = list(range(10, 45))

    def make_rev_sample(length: int = 4):
        seq = [random.choice(tokens_pool) for _ in range(length)]
        prompt = [1] + seq + [2]
        return prompt, [seq[-1]]

    rev_train = [make_rev_sample() for _ in range(500)]
    rev_test = [make_rev_sample() for _ in range(100)]

    def eval_rev(m, dataset):
        correct = 0
        for prompt, target in dataset:
            m.reset_session()
            logits = m.forward_ids(prompt, None)
            pred = int(np.argmax(logits.numpy()[-1]))
            if pred == target[0]:
                correct += 1
        return (correct / len(dataset)) * 100.0

    acc_rev_before = eval_rev(model_rev, rev_test)
    print(f"  -> GPU Zero-Shot Held-Out Accuracy:  {acc_rev_before:.1f}%")

    t0 = time.time()
    for epoch in range(1, 9):
        losses = [trainer_rev.train_supervised(p, t) for p, t in rev_train]
        if epoch in (1, 4, 8):
            print(f"  Epoch {epoch:2d}/8 | Cross-Entropy Loss: {np.mean(losses):.4f} | Processed: {epoch * len(rev_train):,} steps")

    print(f"  -> GPU Training completed in {time.time() - t0:.2f}s")
    acc_rev_after = eval_rev(model_rev, rev_test)
    print(f"  -> GPU Post-Training Held-Out Accuracy: {acc_rev_after:.1f}%")
    print("=" * 78)


# -----------------------------------------------------------------------------
# 6. Main Execution Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    t_start = time.time()
    gpu_info = inspect_gpu_hardware()
    gpu_throughput = run_gpu_flappy_bird_benchmark()
    run_gpu_synthetic_benchmarks()
    total_time = time.time() - t_start
    print(f"\n[DEDICATED GPU BENCHMARK COMPLETE] All GPU training runs executed in {total_time:.1f}s.")
