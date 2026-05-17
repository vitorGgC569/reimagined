"""GRPO vs PPO test (Tese 1, section 'A Morte do Crítico').

Claim under test: GRPO (Group Relative Policy Optimization) matches or
exceeds PPO without needing a value-network critic.  DeepSeek-R1 uses
GRPO to bypass the critic memory cost.

Honest scope:
  * Toy RL task: CartPole-v1.  Continuous-output tasks (LLM-style
    discrete-action policies) would need a much bigger scaffold; the
    no-critic claim is a property of the algorithm, not the task, so
    CartPole is sufficient to validate the direction.
  * Same policy architecture for both.  PPO additionally has a value
    head (the critic); GRPO does not.  This is the variable under
    test: do we lose anything by removing the critic?
  * Metrics:
      - Avg return at end of training (sanity)
      - Total parameters used (the no-critic savings)
      - Wall time (GRPO should be faster per step since no critic
        forward/backward)

We deliberately keep this small (5K env steps, not millions) — the
goal is to validate the algorithm's correctness, not to push CartPole
to perfection.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import set_global_seed


# ──────────────────────────────────────────────────────────────────────
#  Policy / value networks
# ──────────────────────────────────────────────────────────────────────

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x): return self.net(x)


class ValueNet(nn.Module):
    """Critic — used only by PPO."""
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x): return self.net(x).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────
#  Rollout collection
# ──────────────────────────────────────────────────────────────────────

def collect_rollouts(env_make, policy: PolicyNet, n_episodes: int, max_t: int,
                      device: torch.device, gamma: float = 0.99):
    """Sample N episodes; return (obs, actions, rewards, logprobs, returns)."""
    obs_buf, act_buf, lp_buf, ret_buf = [], [], [], []
    rewards_per_ep = []
    for _ in range(n_episodes):
        env = env_make()
        obs, _ = env.reset()
        ep_obs, ep_act, ep_lp, ep_r = [], [], [], []
        for _ in range(max_t):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = policy(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                lp = dist.log_prob(a)
            obs_next, r, term, trunc, _ = env.step(int(a.item()))
            ep_obs.append(obs)
            ep_act.append(int(a.item()))
            ep_lp.append(float(lp.item()))
            ep_r.append(float(r))
            obs = obs_next
            if term or trunc:
                break
        env.close()
        # Compute discounted returns from rewards
        ret, returns = 0.0, []
        for r in reversed(ep_r):
            ret = r + gamma * ret
            returns.append(ret)
        returns.reverse()
        obs_buf.extend(ep_obs); act_buf.extend(ep_act)
        lp_buf.extend(ep_lp); ret_buf.extend(returns)
        rewards_per_ep.append(sum(ep_r))
    return (
        torch.as_tensor(np.array(obs_buf), dtype=torch.float32, device=device),
        torch.as_tensor(act_buf, dtype=torch.long, device=device),
        torch.as_tensor(lp_buf, dtype=torch.float32, device=device),
        torch.as_tensor(ret_buf, dtype=torch.float32, device=device),
        rewards_per_ep,
    )


# ──────────────────────────────────────────────────────────────────────
#  PPO update (with critic)
# ──────────────────────────────────────────────────────────────────────

def ppo_update(policy: PolicyNet, value: ValueNet,
                obs, actions, old_logprobs, returns,
                opt_p: torch.optim.Optimizer, opt_v: torch.optim.Optimizer,
                epochs: int = 4, clip: float = 0.2):
    for _ in range(epochs):
        # Compute value estimates and advantages
        v = value(obs)
        adv = returns - v.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # Policy update with clipped objective
        logits = policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        new_lp = dist.log_prob(actions)
        ratio = (new_lp - old_logprobs).exp()
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
        loss_p = -torch.min(surr1, surr2).mean()
        opt_p.zero_grad(set_to_none=True)
        loss_p.backward()
        opt_p.step()
        # Value update
        loss_v = F.mse_loss(value(obs), returns)
        opt_v.zero_grad(set_to_none=True)
        loss_v.backward()
        opt_v.step()


# ──────────────────────────────────────────────────────────────────────
#  GRPO update (no critic — group-relative advantage)
# ──────────────────────────────────────────────────────────────────────

def grpo_update(policy: PolicyNet,
                 obs, actions, old_logprobs, returns,
                 group_returns_per_episode: List[float],
                 episode_lens: List[int],
                 opt_p: torch.optim.Optimizer,
                 epochs: int = 4, clip: float = 0.2):
    """GRPO substitutes the value-critic baseline with a group-relative
    normalization: each transition's advantage is the EPISODE return
    minus the group (batch-of-episodes) mean, divided by group std.

    Concretely, for a sampled batch of N episodes:
      A_i = (R_i - mean(R)) / std(R)
    where R_i is the total return of episode i.  This advantage is
    broadcast to every step of episode i.
    """
    group_R = torch.as_tensor(group_returns_per_episode, dtype=torch.float32, device=obs.device)
    mu = group_R.mean()
    sd = group_R.std() + 1e-8
    norm_R = (group_R - mu) / sd  # one per episode
    # Broadcast per-episode advantage to each step
    adv_per_step = []
    for ep_idx, ep_len in enumerate(episode_lens):
        adv_per_step.extend([norm_R[ep_idx].item()] * ep_len)
    adv = torch.as_tensor(adv_per_step, dtype=torch.float32, device=obs.device)

    for _ in range(epochs):
        logits = policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        new_lp = dist.log_prob(actions)
        ratio = (new_lp - old_logprobs).exp()
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
        loss = -torch.min(surr1, surr2).mean()
        opt_p.zero_grad(set_to_none=True)
        loss.backward()
        opt_p.step()


# ──────────────────────────────────────────────────────────────────────
#  Training driver
# ──────────────────────────────────────────────────────────────────────

def _episode_lens_from_rewards(rewards_per_ep: List[float], total_steps: int) -> List[int]:
    # We don't carry per-ep lens from collect_rollouts; reconstruct from
    # the obs buffer indirectly.  Actually, the simplest fix: have
    # collect_rollouts return ep_lens directly.  We do this inline below
    # in the run loop to avoid a refactor of the public function.
    raise NotImplementedError("Use the inline variant in the run loop")


def run_ppo(env_make, total_updates: int, episodes_per_update: int, max_t: int,
             device: torch.device, lr: float = 3e-4):
    import time
    env = env_make()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()
    policy = PolicyNet(obs_dim, n_actions).to(device)
    value = ValueNet(obs_dim).to(device)
    opt_p = torch.optim.Adam(policy.parameters(), lr=lr)
    opt_v = torch.optim.Adam(value.parameters(), lr=lr)
    avg_returns: List[float] = []
    t0 = time.time()
    for _ in range(total_updates):
        obs, acts, lps, rets, rew_per_ep = collect_rollouts(env_make, policy, episodes_per_update, max_t, device)
        ppo_update(policy, value, obs, acts, lps, rets, opt_p, opt_v)
        avg_returns.append(float(np.mean(rew_per_ep)))
    return {
        "final_avg_return": avg_returns[-1] if avg_returns else float("nan"),
        "best_avg_return": max(avg_returns) if avg_returns else float("nan"),
        "all_avg_returns": avg_returns,
        "wall_time_s": time.time() - t0,
        "total_params": sum(p.numel() for p in list(policy.parameters()) + list(value.parameters())),
        "policy_params": sum(p.numel() for p in policy.parameters()),
        "value_params": sum(p.numel() for p in value.parameters()),
    }


def run_grpo(env_make, total_updates: int, episodes_per_update: int, max_t: int,
              device: torch.device, lr: float = 3e-4):
    import time
    env = env_make()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()
    policy = PolicyNet(obs_dim, n_actions).to(device)
    opt_p = torch.optim.Adam(policy.parameters(), lr=lr)
    avg_returns: List[float] = []
    t0 = time.time()
    for _ in range(total_updates):
        # Collect with episode-length tracking
        obs_buf, act_buf, lp_buf, ret_buf, rew_per_ep, ep_lens = _collect_with_lens(
            env_make, policy, episodes_per_update, max_t, device,
        )
        grpo_update(policy, obs_buf, act_buf, lp_buf, ret_buf,
                    group_returns_per_episode=rew_per_ep,
                    episode_lens=ep_lens, opt_p=opt_p)
        avg_returns.append(float(np.mean(rew_per_ep)))
    return {
        "final_avg_return": avg_returns[-1] if avg_returns else float("nan"),
        "best_avg_return": max(avg_returns) if avg_returns else float("nan"),
        "all_avg_returns": avg_returns,
        "wall_time_s": time.time() - t0,
        "total_params": sum(p.numel() for p in policy.parameters()),
        "policy_params": sum(p.numel() for p in policy.parameters()),
        "value_params": 0,
    }


def _collect_with_lens(env_make, policy, n_episodes, max_t, device, gamma=0.99):
    """Variant of collect_rollouts that also returns per-episode lengths."""
    obs_buf, act_buf, lp_buf, ret_buf = [], [], [], []
    rewards_per_ep, ep_lens = [], []
    for _ in range(n_episodes):
        env = env_make()
        obs, _ = env.reset()
        ep_obs, ep_act, ep_lp, ep_r = [], [], [], []
        for _ in range(max_t):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = policy(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                lp = dist.log_prob(a)
            obs_next, r, term, trunc, _ = env.step(int(a.item()))
            ep_obs.append(obs); ep_act.append(int(a.item()))
            ep_lp.append(float(lp.item())); ep_r.append(float(r))
            obs = obs_next
            if term or trunc:
                break
        env.close()
        ret, returns = 0.0, []
        for r in reversed(ep_r):
            ret = r + gamma * ret
            returns.append(ret)
        returns.reverse()
        obs_buf.extend(ep_obs); act_buf.extend(ep_act)
        lp_buf.extend(ep_lp); ret_buf.extend(returns)
        rewards_per_ep.append(sum(ep_r))
        ep_lens.append(len(ep_r))
    return (
        torch.as_tensor(np.array(obs_buf), dtype=torch.float32, device=device),
        torch.as_tensor(act_buf, dtype=torch.long, device=device),
        torch.as_tensor(lp_buf, dtype=torch.float32, device=device),
        torch.as_tensor(ret_buf, dtype=torch.float32, device=device),
        rewards_per_ep,
        ep_lens,
    )


# ──────────────────────────────────────────────────────────────────────
#  Test entry
# ──────────────────────────────────────────────────────────────────────

def test_grpo_vs_ppo(*, device: torch.device, seed: int = 42, total_updates: int = 50,
                      episodes_per_update: int = 8, max_t: int = 200) -> dict:
    try:
        import gymnasium as gym
    except ImportError:
        return {
            "ERROR": "gymnasium not installed.  pip install gymnasium",
        }

    def env_make():
        env = gym.make("CartPole-v1")
        env.reset(seed=seed)
        return env

    set_global_seed(seed)
    ppo_result = run_ppo(env_make, total_updates, episodes_per_update, max_t, device)
    set_global_seed(seed)
    grpo_result = run_grpo(env_make, total_updates, episodes_per_update, max_t, device)

    return {
        "PPO (with critic)": ppo_result,
        "GRPO (no critic)": grpo_result,
    }


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="GRPO vs PPO test on CartPole.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--updates", type=int, default=50)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--max_t", type=int, default=200)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rl_grpo] device={device.type}")
    results = test_grpo_vs_ppo(device=device, seed=args.seed,
                                total_updates=args.updates,
                                episodes_per_update=args.episodes,
                                max_t=args.max_t)
    print(f"\n=== test_grpo_vs_ppo (CartPole-v1) ===")
    for label, val in results.items():
        if not isinstance(val, dict):
            print(f"  {label}: {val}")
            continue
        print(f"  {label}:")
        for k, v in val.items():
            if k == "all_avg_returns":
                continue
            if isinstance(v, float):
                print(f"    {k:<32} {v:.3f}")
            else:
                print(f"    {k:<32} {v}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        serializable = {}
        for label, val in results.items():
            if isinstance(val, dict):
                serializable[label] = {k: v for k, v in val.items() if k != "all_avg_returns"}
            else:
                serializable[label] = val
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
