"""
tests/test_fullstack_scaled.py

Teste dedicado do Mamba-3 Full Stack escalado:
  - 8 camadas (espaço para cada mecanismo operar sem competição)
  - d_model=512 (experts com dimensionalidade suficiente)
  - Todas as camadas juntas: Mamba-3 + Atenção + MoE + KAN

Layout das 8 camadas:
  L1: Mamba-3 + MoE  (period=2, slot=0)
  L2: Mamba-3 + KAN  (use_kan=True, sem MoE)
  L3: Mamba-3 + MoE
  L4: Mamba-3 + Atenção GQA + KAN  (period=4, slot=3)
  L5: Mamba-3 + MoE
  L6: Mamba-3 + KAN
  L7: Mamba-3 + MoE
  L8: Mamba-3 + Atenção GQA + KAN

Comparativo direto contra o Full Stack pequeno (4 camadas, d_model=128)
para quantificar o ganho de escala.

Hardware: AMD Radeon RX 7600 (gfx1102, 8 GB VRAM)
"""
from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
HIP_BUILD = REPO_ROOT / "OXN" / "nsos" / "build-gm-hip"
sys.path.insert(0, str(HIP_BUILD))

import nsos_ext as nsos  # noqa: E402
import numpy as np  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    except Exception:
        pass

os.environ["NSOS_MAMBA3_GPU_PROVIDER"]        = "parallel_fp32_v1"
os.environ["NSOS_MAMBA3_PROJECTION_PROVIDER"] = "exact_fp32"

# ── Parâmetros do benchmark ────────────────────────────────────────────────
N_FLAPPY = 6000
N_KV_TR  = 750
N_KV_TE  = 100
N_REV_TR = 600
N_REV_TE = 100
LR       = 0.002

CONFIGS = [
    # (label, num_layers, d_model, n_heads)
    ("Full Stack  4L d128",  4,  128, 4),
    ("Full Stack  8L d256",  8,  256, 8),
    ("Full Stack  8L d512",  8,  512, 8),
]


# ── Ambientes ──────────────────────────────────────────────────────────────
class Pipe:
    def __init__(self, x, top_height, gap=110.0, width=50.0):
        self.x = x; self.top_height = top_height
        self.gap = gap; self.width = width; self.passed = False

    @property
    def bottom_y(self): return self.top_height + self.gap
    @property
    def gap_center(self): return self.top_height + self.gap / 2.0


class FlappyEnvironment:
    WIDTH = 300; HEIGHT = 400; GRAVITY = 0.9; FLAP_IMPULSE = -7.5
    MAX_VELOCITY = 9.0; PIPE_SPEED = 3.0; PIPE_SPAWN_DIST = 140.0
    BIRD_X = 50.0; BIRD_RADIUS = 12.0

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.bird_y = float(self.HEIGHT // 2); self.bird_vel = 0.0
        self.pipes = [Pipe(200, 120), Pipe(340, 160), Pipe(480, 100)]
        self.score = 0; self.frames = 0; self.game_over = False
        return self._obs()

    def _spawn(self, x):
        self.pipes.append(Pipe(x, self.rng.uniform(50, self.HEIGHT - 170)))

    def _next_pipe(self):
        for p in self.pipes:
            if p.x + p.width >= self.BIRD_X - self.BIRD_RADIUS:
                return p
        return self.pipes[0]

    def _obs(self):
        p = self._next_pipe()
        return (p.x + p.width / 2 - self.BIRD_X, p.gap_center - self.bird_y, self.bird_vel)

    def step(self, action):
        if self.game_over:
            return self._obs(), 0.0, True, self.score
        self.frames += 1
        if action == 1: self.bird_vel = self.FLAP_IMPULSE
        self.bird_vel = min(self.bird_vel + self.GRAVITY, self.MAX_VELOCITY)
        self.bird_y += self.bird_vel
        for p in self.pipes:
            p.x -= self.PIPE_SPEED
            if not p.passed and p.x + p.width < self.BIRD_X:
                p.passed = True; self.score += 1
        if self.pipes and self.pipes[0].x + self.pipes[0].width < 0:
            self.pipes.pop(0); self._spawn(self.pipes[-1].x + self.PIPE_SPAWN_DIST)
        if self.bird_y - self.BIRD_RADIUS <= 0 or self.bird_y + self.BIRD_RADIUS >= self.HEIGHT:
            self.game_over = True
        np_ = self._next_pipe()
        if (np_.x <= self.BIRD_X + self.BIRD_RADIUS and np_.x + np_.width >= self.BIRD_X - self.BIRD_RADIUS):
            if self.bird_y - self.BIRD_RADIUS < np_.top_height or self.bird_y + self.BIRD_RADIUS > np_.bottom_y:
                self.game_over = True
        reward = 1.0
        if self.game_over: reward = -20.0
        elif action == 1 and abs(self.bird_y - np_.gap_center) < 30: reward += 0.5
        return self._obs(), reward, self.game_over, self.score


class FlappyTok:
    VS = 64; BOS = 1; SEP = 2; DX = 10; DY = 25; VEL = 48; COAST = 60; FLAP = 61

    @classmethod
    def enc(cls, obs):
        dx, dy, vel = obs
        b_dx  = cls.DX  + min(9,  int(max(0, min(200, dx))  / 20.0))
        b_dy  = cls.DY  + int(max(0, min(1, (dy + 150) / 300)) * 19.99)
        b_vel = cls.VEL + int(max(0, min(1, (vel + 8) / 18)) * 9.99)
        return [cls.BOS, b_dx, b_dy, b_vel, cls.SEP]


def expert(obs):
    dx, dy, vel = obs
    pred = dy - vel * 3.5
    if pred > 12: return 1
    if pred < -20: return 0
    return 1 if vel > 2 else 0


# ── Construção do modelo Full Stack ───────────────────────────────────────
def build_full_stack(num_layers: int, d_model: int, n_heads: int) -> Tuple[nsos.JambaModel, int]:
    cfg = nsos.ModelConfig()
    cfg.architecture_schema_version = 3
    cfg.num_layers        = num_layers
    cfg.d_model           = d_model
    cfg.n_heads           = n_heads
    cfg.n_kv_heads        = max(1, n_heads // 2)
    cfg.vocab_size        = FlappyTok.VS
    cfg.mamba3_enabled    = True
    cfg.mamba3_schema_version = 1
    # Atenção a cada 4 camadas (slot 3 → camadas 4, 8, 12…)
    cfg.attention_period  = 4
    cfg.attention_slot    = 3
    # MoE nas camadas ímpares (slot 0 → camadas 1, 3, 5, 7…)
    cfg.use_moe               = True
    cfg.num_experts           = 4
    cfg.num_experts_per_token = 2
    cfg.moe_period            = 2
    cfg.moe_slot              = 0
    # KAN nas camadas sem MoE (o runtime aplica onde is_moe=False)
    cfg.use_kan = True
    model = nsos.JambaModel(cfg, nsos.Device.GPU)
    params = sum(p.data.size for p in model.parameters())
    return model, params


# ── Runners ───────────────────────────────────────────────────────────────
def train_flappy(label, model, dataset):
    print(f"\n--- Flappy Bird: {label} ---", flush=True)
    trainer = nsos.Trainer(model, LR)
    trainer.weight_decay = 0.0001
    trainer.max_grad_norm = 1.0
    losses = []
    t0 = time.time()
    for i, (prompt, target) in enumerate(dataset, 1):
        loss = trainer.train_supervised(prompt, target)
        losses.append(loss)
        if i in (1, len(dataset) // 2, len(dataset)):
            print(f"  Step {i}/{len(dataset)} | loss={loss:.4f} avg={np.mean(losses[-50:]):.4f}", flush=True)
    elapsed = time.time() - t0
    sps = len(dataset) / max(0.001, elapsed)
    print(f"  Concluido: {elapsed:.1f}s  {sps:.1f} st/s  loss_final={np.mean(losses[-30:]):.4f}", flush=True)

    # 10 voos de teste
    model.set_training_mode(False)
    env = FlappyEnvironment(seed=555)
    scores = []
    for _ in range(10):
        model.reset_session()
        obs = env.reset(); done = False
        while not done and env.frames < 2000:
            logits = model.forward_ids(FlappyTok.enc(obs), None)
            lnp = logits.numpy()[-1]
            act = 1 if float(lnp[FlappyTok.FLAP]) > float(lnp[FlappyTok.COAST]) else 0
            obs, _, done, sc = env.step(act)
        scores.append(sc)
    print(f"  Voo: avg={np.mean(scores):.1f} max={max(scores)} pipes", flush=True)
    return {"sps": sps, "loss": float(np.mean(losses[-30:])), "avg_pipes": float(np.mean(scores)), "max_pipes": max(scores)}


def train_kv(label, model, train_data, test_data):
    print(f"\n--- KV Recall: {label} ---", flush=True)

    def acc(m, data):
        m.set_training_mode(False)
        ok = 0
        for seq, tgt in data:
            m.reset_session()
            logits = m.forward_ids(seq, None)
            if int(np.argmax(logits.numpy()[-1])) == tgt[0]:
                ok += 1
        return 100.0 * ok / len(data)

    before = acc(model, test_data)
    model.set_training_mode(True)
    trainer = nsos.Trainer(model, LR); trainer.max_grad_norm = 1.0
    t0 = time.time(); losses = []
    for i, (p, t) in enumerate(train_data, 1):
        losses.append(trainer.train_supervised(p, t))
        if i in (1, len(train_data) // 2, len(train_data)):
            print(f"  Step {i}/{len(train_data)} | loss={losses[-1]:.4f}", flush=True)
    after = acc(model, test_data)
    print(f"  Acc: {before:.1f}% -> {after:.1f}%  ({time.time()-t0:.1f}s)", flush=True)
    return {"before": before, "after": after}


def train_rev(label, model, train_data, test_data):
    print(f"\n--- Reversal: {label} ---", flush=True)

    def acc(m, data):
        m.set_training_mode(False)
        ok = 0
        for seq, tgt in data:
            m.reset_session()
            logits = m.forward_ids(seq, None)
            if int(np.argmax(logits.numpy()[-1])) == tgt[0]:
                ok += 1
        return 100.0 * ok / len(data)

    before = acc(model, test_data)
    model.set_training_mode(True)
    trainer = nsos.Trainer(model, LR); trainer.max_grad_norm = 1.0
    t0 = time.time(); losses = []
    for i, (p, t) in enumerate(train_data, 1):
        losses.append(trainer.train_supervised(p, t))
        if i in (1, len(train_data) // 2, len(train_data)):
            print(f"  Step {i}/{len(train_data)} | loss={losses[-1]:.4f}", flush=True)
    after = acc(model, test_data)
    print(f"  Acc: {before:.1f}% -> {after:.1f}%  ({time.time()-t0:.1f}s)", flush=True)
    return {"before": before, "after": after}


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 78, flush=True)
    print("   MAMBA-3 FULL STACK — VALIDAÇÃO DE ESCALA (RX 7600 / gfx1102)", flush=True)
    print("   Testando: 4L d128  vs  8L d256  vs  8L d512", flush=True)
    print("=" * 78, flush=True)

    # Datasets comuns
    print("\n[Gerando datasets...]", flush=True)
    env = FlappyEnvironment(seed=777)
    flappy: List[Tuple[List[int], List[int]]] = []
    while len(flappy) < N_FLAPPY:
        obs = env.reset(); done = False
        while not done and len(flappy) < N_FLAPPY:
            act = expert(obs)
            flappy.append((FlappyTok.enc(obs), [FlappyTok.FLAP if act == 1 else FlappyTok.COAST]))
            obs, _, done, _ = env.step(act)

    keys = list(range(10, 30)); vals = list(range(35, 55))
    rng = random.Random(42)
    def mk_kv():
        sk = rng.sample(keys, 4); sv = [rng.choice(vals) for _ in range(4)]
        seq = []
        for k, v in zip(sk, sv): seq.extend([k, v])
        idx = rng.randint(0, 3)
        return seq + [sk[idx]], [sv[idx]]
    kv_tr = [mk_kv() for _ in range(N_KV_TR)]
    kv_te = [mk_kv() for _ in range(N_KV_TE)]

    pool = list(range(10, 45)); rng2 = random.Random(88)
    def mk_rev():
        s = [rng2.choice(pool) for _ in range(4)]
        return [1] + s + [2], [s[-1]]
    rev_tr = [mk_rev() for _ in range(N_REV_TR)]
    rev_te = [mk_rev() for _ in range(N_REV_TE)]
    print(f"  Flappy {N_FLAPPY} | KV {N_KV_TR}/{N_KV_TE} | Rev {N_REV_TR}/{N_REV_TE}", flush=True)

    results = {}
    for label, nl, dm, nh in CONFIGS:
        print(f"\n{'=' * 78}", flush=True)
        print(f"  CONFIG: {label}  (nl={nl} dm={dm} nh={nh})", flush=True)
        print(f"{'=' * 78}", flush=True)

        m_fl, params = build_full_stack(nl, dm, nh)
        vram_mb = params * 4 / 1024 / 1024
        print(f"  Params: {params:,}  Weights: {vram_mb:.1f} MB  (~{vram_mb*3:.1f} MB c/grads)", flush=True)

        r_fl  = train_flappy(label, m_fl,  list(flappy))
        del m_fl

        m_kv, _ = build_full_stack(nl, dm, nh)
        r_kv  = train_kv(label, m_kv,  kv_tr, kv_te)
        del m_kv

        m_rv, _ = build_full_stack(nl, dm, nh)
        r_rev = train_rev(label, m_rv, rev_tr, rev_te)
        del m_rv

        results[label] = {"params": params, "flappy": r_fl, "kv": r_kv, "rev": r_rev}

    # Scorecard
    print("\n" + "=" * 78, flush=True)
    print("   SCORECARD FINAL — FULL STACK EM ESCALA", flush=True)
    print("=" * 78, flush=True)
    hdr = f"{'Config':<22} | {'Params':>10} | {'st/s':>7} | {'KV':>7} | {'Reversal':>9}"
    print(hdr, flush=True)
    print("-" * 78, flush=True)
    for label, r in results.items():
        print(
            f"{label:<22} | {r['params']:>10,} | {r['flappy']['sps']:>7.1f} | "
            f"{r['kv']['after']:>6.1f}% | {r['rev']['after']:>8.1f}%",
            flush=True,
        )
    print("=" * 78, flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[CONCLUIDO] Total: {time.time()-t0:.1f}s", flush=True)
