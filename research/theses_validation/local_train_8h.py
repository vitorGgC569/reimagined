"""8-hour local baseline run on the user's GTX 1050 Ti.

Trains a small Transformer LM (~5M params, 6 layers, d_model=256) on
WikiText-2-raw-v1 char-level for 8 hours wall-clock.  Produces our first
real baseline for VISION.md Eixo 2 (training quality):
  - WikiText-2 held-out PPL
  - Sample generations
  - Loss curve + eval curve
  - Checkpoint every 30 min so a crash doesn't kill the run

Design choices:
  - Char-level tokenization keeps vocab tiny (~70) and eliminates BPE
    setup time; this is a BASELINE, not a SOTA attempt.
  - Manual scaled-dot-product attention (NOT torch.nn.functional's
    scaled_dot_product_attention) because that triggers a segfault
    on GTX 1050 Ti (sm_61, Pascal) with torch 2.6 on Windows —
    reproduced earlier in the validation harness.  Math attention
    is slower but reliable.
  - No torch.compile, no mixed precision — sm_61 doesn't support
    BF16 Tensor Cores anyway, and torch.compile on Pascal is hit or miss.
  - AdamW + cosine schedule with 200 warmup steps.  Standard, no exotica.

Outputs (relative to repo root):
  research/theses_validation/results/local_8h/
    config.json              — frozen hyperparams + git SHA
    run.json                 — final summary (PPL, params, wall time, etc.)
    train_curve.json         — every-step loss
    eval_curve.json          — every-Neval-step held-out loss
    checkpoints/step_NNNN.pt — model state_dict at 30-min intervals
    final_checkpoint.pt      — last state_dict (end-of-run)
    samples.txt              — 5 sample generations from final model
    summary.md               — human-readable report
"""
from __future__ import annotations

# Eager datasets import BEFORE torch — Windows segfault workaround
# (see research/theses_validation/common/data.py for the original diagnosis).
import datasets  # noqa: F401

import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
#  Config — frozen at start of run, written to config.json
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Model
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    dropout: float = 0.0
    max_seq: int = 192          # > seq_len, with margin for inference

    # Data
    seq_len: int = 128
    batch_size: int = 24
    # Cap WikiText slice sizes (train is huge; we use a sub-slice that
    # fits in RAM cleanly and still gives multi-epoch coverage in 8 h).
    train_max_chars: int = 4_000_000   # ~4M chars
    eval_max_chars: int = 250_000

    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 200
    min_lr_scale: float = 0.1    # cosine floor

    # Schedule
    wall_time_budget_s: int = 8 * 60 * 60     # 8 hours
    checkpoint_every_s: int = 30 * 60         # 30 min
    eval_every_steps: int = 1000
    eval_n_batches: int = 32
    log_every_steps: int = 50

    seed: int = 42


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=10", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


# ──────────────────────────────────────────────────────────────────────
#  Data loader (same logic as common.data but inlined for clarity)
# ──────────────────────────────────────────────────────────────────────

def load_data(cfg: Config):
    print(f"[data] loading WikiText-2-raw-v1...", flush=True)
    ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    rows = ds["text"]

    def clean(start: int, n_rows: int, max_chars: int) -> str:
        chunk = " ".join(line for line in rows[start:start + n_rows] if len(line.strip()) > 10)
        chunk = chunk.lower()
        chunk = re.sub(r"[^a-z0-9 \.,;!?']", "", chunk)
        chunk = re.sub(r"\s+", " ", chunk)
        return chunk[:max_chars]

    train_text = clean(0, 30000, cfg.train_max_chars)
    eval_text = clean(30000, 5000, cfg.eval_max_chars)

    vocab_chars = sorted(set(train_text))
    char_to_ix = {c: i for i, c in enumerate(vocab_chars)}
    unk_ix = len(vocab_chars)
    ix_to_char = {i: c for c, i in char_to_ix.items()}
    ix_to_char[unk_ix] = "?"
    vocab_size = len(vocab_chars) + 1  # +1 unk

    train_ids = torch.tensor([char_to_ix[c] for c in train_text], dtype=torch.long)
    eval_ids = torch.tensor([char_to_ix.get(c, unk_ix) for c in eval_text], dtype=torch.long)

    print(f"[data] vocab={vocab_size} train_chars={train_ids.numel():,} "
          f"eval_chars={eval_ids.numel():,}", flush=True)
    return train_ids, eval_ids, vocab_size, char_to_ix, ix_to_char


def get_batch(data: torch.Tensor, cfg: Config, device: torch.device,
              rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(0, data.size(0) - cfg.seq_len - 1, (cfg.batch_size,), generator=rng)
    x = torch.stack([data[i:i + cfg.seq_len] for i in ix]).to(device, non_blocking=True)
    y = torch.stack([data[i + 1:i + cfg.seq_len + 1] for i in ix]).to(device, non_blocking=True)
    return x, y


# ──────────────────────────────────────────────────────────────────────
#  Model
# ──────────────────────────────────────────────────────────────────────

class ManualCausalAttention(nn.Module):
    """Plain scaled-dot-product attention with explicit causal mask.

    We avoid torch.nn.functional.scaled_dot_product_attention because it
    triggers a segfault on GTX 1050 Ti (sm_61, Pascal) with torch 2.6
    on Windows.  Reproduced in the validation harness; switching to
    manual attention works around the kernel selection issue.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(d, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)  # (b, h, t, hd)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        # Attention scores: (b, h, t, t)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # Causal mask: lower triangular True (keep), upper True (mask)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)              # (b, h, t, hd)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = ManualCausalAttention(cfg.d_model, cfg.n_heads)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class SmallLM(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size)
        # Tie weights — matches GPT-2 convention, gives free param savings
        self.head.weight = self.tok.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        assert t <= self.cfg.max_seq
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, -1)
        h = self.tok(x) + self.pos(pos)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln_f(h))


# ──────────────────────────────────────────────────────────────────────
#  LR schedule
# ──────────────────────────────────────────────────────────────────────

def cosine_lr(step: int, total: int, cfg: Config) -> float:
    """Cosine with linear warmup.  Total is estimated from observed
    step rate after warmup."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    decay_span = max(total - cfg.warmup_steps, 1)
    progress = min((step - cfg.warmup_steps) / decay_span, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * cfg.min_lr_scale + cfg.lr * (1.0 - cfg.min_lr_scale) * cosine


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


# ──────────────────────────────────────────────────────────────────────
#  Eval + generation
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_loss(model: SmallLM, eval_data: torch.Tensor, cfg: Config,
              device: torch.device) -> float:
    model.eval()
    rng = torch.Generator()
    rng.manual_seed(7777)  # fixed across calls so eval noise is constant
    total = 0.0
    for _ in range(cfg.eval_n_batches):
        x, y = get_batch(eval_data, cfg, device, rng)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total += float(loss.item())
    model.train()
    return total / cfg.eval_n_batches


@torch.no_grad()
def sample(model: SmallLM, prompt: str, ix_to_char: dict, char_to_ix: dict,
           cfg: Config, device: torch.device, max_new: int = 200,
           temperature: float = 0.8, top_k: int = 40) -> str:
    model.eval()
    # Encode prompt; unknown chars become a benign filler (space).
    ids = [char_to_ix.get(c, char_to_ix.get(" ", 0)) for c in prompt.lower()]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    out = prompt
    for _ in range(max_new):
        x_in = x[:, -cfg.max_seq:]
        logits = model(x_in)[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_ix = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_ix], dim=1)
        out += ix_to_char.get(int(next_ix.item()), "?")
    model.train()
    return out


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = Config()
    # Allow CLI override of wall time budget for smoke testing
    if "--minutes" in sys.argv:
        idx = sys.argv.index("--minutes")
        cfg.wall_time_budget_s = int(float(sys.argv[idx + 1]) * 60)
        cfg.checkpoint_every_s = max(cfg.checkpoint_every_s, cfg.wall_time_budget_s // 4)
    if "--quick" in sys.argv:
        cfg.wall_time_budget_s = 5 * 60         # 5 min smoke
        cfg.eval_every_steps = 200
        cfg.checkpoint_every_s = 120
        cfg.log_every_steps = 10

    out_dir = Path(__file__).resolve().parent / "results" / "local_8h"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    # Seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[boot] device={device.type} torch={torch.__version__} "
          f"cuda_avail={torch.cuda.is_available()}", flush=True)
    if device.type == "cuda":
        print(f"[boot] gpu={torch.cuda.get_device_name(0)} "
              f"mem={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    train_data, eval_data, vocab_size, char_to_ix, ix_to_char = load_data(cfg)

    model = SmallLM(cfg, vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] params={n_params:,} d_model={cfg.d_model} layers={cfg.n_layers} "
          f"heads={cfg.n_heads}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                   betas=(0.9, 0.95))

    # Persist config + git SHA up front so the artifact is self-describing.
    config_blob = {
        **asdict(cfg),
        "vocab_size": vocab_size,
        "total_params": n_params,
        "git_sha": _git_sha(),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (out_dir / "config.json").write_text(json.dumps(config_blob, indent=2))

    # Estimate total steps for cosine schedule.  We use a placeholder
    # (will update after warmup measures real step time).
    train_curve: list[tuple[int, float, float]] = []   # (step, lr, loss)
    eval_curve: list[tuple[int, float]] = []           # (step, eval_loss)
    est_total_steps = 200_000  # initial; corrected after step 100

    data_rng = torch.Generator()
    data_rng.manual_seed(cfg.seed * 7)

    print(f"[run] wall budget = {cfg.wall_time_budget_s}s ({cfg.wall_time_budget_s / 3600:.2f} h)",
          flush=True)
    print(f"[run] checkpoint every {cfg.checkpoint_every_s}s, eval every {cfg.eval_every_steps} steps",
          flush=True)

    t_start = time.time()
    t_last_checkpoint = t_start
    t_warmup_done = None
    step = 0

    try:
        while True:
            now = time.time()
            elapsed = now - t_start
            if elapsed >= cfg.wall_time_budget_s:
                print(f"[run] wall time budget reached ({elapsed:.0f}s); stopping.", flush=True)
                break

            step += 1
            lr = cosine_lr(step, est_total_steps, cfg)
            set_lr(optimizer, lr)

            x, y = get_batch(train_data, cfg, device, data_rng)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            loss_v = float(loss.detach().item())
            train_curve.append((step, lr, loss_v))

            # Recalibrate total-step estimate once warmup is done
            if step == cfg.warmup_steps:
                t_warmup_done = now
                elapsed_warmup = now - t_start
                step_rate = step / elapsed_warmup
                est_total_steps = int(step_rate * cfg.wall_time_budget_s)
                print(f"[run] recalibrated: {step_rate:.2f} steps/s, "
                      f"est_total_steps={est_total_steps:,}", flush=True)

            if step % cfg.log_every_steps == 0:
                rate = step / max(elapsed, 1e-6)
                eta_s = max(cfg.wall_time_budget_s - elapsed, 0)
                eta_h = eta_s / 3600
                tok_s = rate * cfg.batch_size * cfg.seq_len
                print(f"[step {step:>7d}] loss={loss_v:.4f} lr={lr:.2e} "
                      f"{rate:.2f} steps/s ({tok_s:.0f} tok/s) eta {eta_h:.2f}h",
                      flush=True)

            if step % cfg.eval_every_steps == 0:
                ev = eval_loss(model, eval_data, cfg, device)
                eval_curve.append((step, ev))
                ppl = math.exp(ev)
                print(f"[eval  step={step}] held-out loss={ev:.4f} ppl={ppl:.2f}",
                      flush=True)
                # Persist curves periodically so a crash doesn't lose them.
                (out_dir / "train_curve.json").write_text(
                    json.dumps(train_curve, indent=1)
                )
                (out_dir / "eval_curve.json").write_text(
                    json.dumps(eval_curve, indent=1)
                )

            if now - t_last_checkpoint >= cfg.checkpoint_every_s:
                ckpt_path = out_dir / "checkpoints" / f"step_{step:07d}.pt"
                torch.save({
                    "step": step,
                    "model_state": model.state_dict(),
                    "config": asdict(cfg),
                    "vocab": {"char_to_ix": char_to_ix, "ix_to_char": ix_to_char,
                              "vocab_size": vocab_size},
                    "elapsed_s": elapsed,
                    "loss": loss_v,
                }, ckpt_path)
                t_last_checkpoint = now
                print(f"[ckpt] saved {ckpt_path.name}", flush=True)

    except KeyboardInterrupt:
        print("\n[run] interrupted — saving final state then exiting.", flush=True)

    # Final eval + checkpoint
    elapsed = time.time() - t_start
    print(f"\n[final] total elapsed: {elapsed:.0f}s ({elapsed / 3600:.2f}h), {step:,} steps",
          flush=True)
    final_eval = eval_loss(model, eval_data, cfg, device)
    final_ppl = math.exp(final_eval)
    print(f"[final] held-out loss={final_eval:.4f} ppl={final_ppl:.2f}", flush=True)

    final_path = out_dir / "final_checkpoint.pt"
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "config": asdict(cfg),
        "vocab": {"char_to_ix": char_to_ix, "ix_to_char": ix_to_char, "vocab_size": vocab_size},
        "elapsed_s": elapsed,
        "final_eval_loss": final_eval,
        "final_eval_ppl": final_ppl,
    }, final_path)
    print(f"[final] checkpoint -> {final_path}", flush=True)

    # Sample generations
    prompts = [
        "the history of ",
        "in the year ",
        "she opened the door and ",
        "the most important thing about ",
        "scientists believe that ",
    ]
    sample_blob = []
    for p in prompts:
        gen = sample(model, p, ix_to_char, char_to_ix, cfg, device, max_new=200)
        sample_blob.append({"prompt": p, "generation": gen})
        print(f"\n>>> {p}\n{gen}\n", flush=True)
    (out_dir / "samples.txt").write_text(
        "\n\n---\n\n".join(f"PROMPT: {s['prompt']}\n\n{s['generation']}" for s in sample_blob),
        encoding="utf-8",
    )

    # Run summary
    summary = {
        "git_sha": config_blob["git_sha"],
        "gpu": config_blob["gpu"],
        "total_params": n_params,
        "vocab_size": vocab_size,
        "wall_time_s": elapsed,
        "wall_time_h": elapsed / 3600,
        "total_steps": step,
        "tokens_seen": step * cfg.batch_size * cfg.seq_len,
        "final_train_loss": sum(l for _, _, l in train_curve[-50:]) / max(len(train_curve[-50:]), 1),
        "final_eval_loss": final_eval,
        "final_eval_ppl": final_ppl,
        "final_steps_per_s": step / elapsed if elapsed > 0 else 0,
        "samples": sample_blob,
    }
    (out_dir / "run.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out_dir / "train_curve.json").write_text(json.dumps(train_curve, indent=1))
    (out_dir / "eval_curve.json").write_text(json.dumps(eval_curve, indent=1))

    # Markdown summary for humans
    md = [
        f"# Local 8 h baseline run — summary",
        f"",
        f"- **GPU:** {config_blob['gpu']}",
        f"- **Git SHA:** {config_blob['git_sha']}",
        f"- **Total params:** {n_params:,}",
        f"- **Vocab size:** {vocab_size}",
        f"- **Wall time:** {elapsed:.0f} s ({elapsed/3600:.2f} h)",
        f"- **Total steps:** {step:,}",
        f"- **Tokens seen:** {summary['tokens_seen']:,}",
        f"- **Steady-state steps/s:** {summary['final_steps_per_s']:.2f}",
        f"",
        f"## Results",
        f"",
        f"| metric | value |",
        f"|---|---|",
        f"| final train loss (last 50 steps avg) | {summary['final_train_loss']:.4f} |",
        f"| final held-out loss | {final_eval:.4f} |",
        f"| **final held-out PPL (char-level)** | **{final_ppl:.2f}** |",
        f"",
        f"For reference, GPT-2-small (124M) on WikiText-2 (BPE) is ~29 PPL.",
        f"Char-level PPL is on a different scale (smaller vocab → lower PPL is",
        f"easier), so the absolute number isn't directly comparable; the value",
        f"of this run is as a BASELINE against which future runs are measured.",
        f"",
        f"## Sample generations",
        f"",
    ]
    for s in sample_blob:
        md.append(f"### Prompt: `{s['prompt']}`")
        md.append("")
        md.append("```")
        md.append(s["generation"])
        md.append("```")
        md.append("")
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[done] artifacts in {out_dir}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
