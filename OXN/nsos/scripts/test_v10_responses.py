"""
Batch inference test for NSOS v10_gpu.
Runs a fixed list of prompts across multiple task kinds, prints results
with metrics so we can judge whether the model is actually responding.

Uses phase6_memory.bin — the most recent checkpoint (state after all 4
real training phases + 3 evals).  Phases 5/6 had 0 training steps so
this is functionally identical to phase4_instructions.bin.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

# UTF-8 safe stdout on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Build dir & model paths ─────────────────────────────────────────────
WORKTREE  = Path(__file__).resolve().parent.parent.parent.parent
BUILD_DIR = Path("C:/Users/Oxta/Desktop/reimagined-main/.claude/worktrees/"
                 "clever-roentgen-ba007c/OXN/nsos/build-cuda-validation")
RUN_DIR   = WORKTREE / "OXN/nsos/scripts/live_distill_v10_gpu"
MODEL_BIN = RUN_DIR / "phase6_memory.bin"
MODEL_CFG = RUN_DIR / "effective_model_config.json"

# Load nsos_ext + CUDA DLLs
if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))
if os.name == "nt":
    for p in [BUILD_DIR,
              Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin")]:
        try:
            if p.exists():
                os.add_dll_directory(str(p))
        except (AttributeError, OSError):
            pass

import nsos_ext as nsos  # type: ignore  # noqa: E402

# ── Load model ──────────────────────────────────────────────────────────
config = nsos.ModelConfig()
if MODEL_CFG.exists():
    for k, v in json.loads(MODEL_CFG.read_text("utf-8")).items():
        if hasattr(config, k):
            setattr(config, k, v)

# Force CPU for inference to keep PSU envelope safe (recent shutdowns
# happened on GPU during sustained loads).  Inference is single-stream
# anyway so the GPU speedup matters less here.
config.use_cuda = False

print(f"[load] {MODEL_BIN.name} (size={MODEL_BIN.stat().st_size/1e6:.0f} MB)")
print(f"[load] vocab={config.vocab_size} layers={config.num_layers} d_model={config.d_model}")

engine = nsos.InferenceEngine()
t0 = time.time()
if not engine.load_model(str(MODEL_BIN), config):
    print(f"[ERROR] Failed to load model", file=sys.stderr)
    sys.exit(1)
print(f"[load] loaded in {time.time()-t0:.1f}s\n")

opts = nsos.GenerationOptions()
opts.max_context_tokens = 512
opts.max_tokens         = 32     # short — we just want signal, not novels
opts.temperature        = 0.4
opts.top_p              = 0.85
opts.top_k              = 20
opts.stream             = False


def build_prompt(kind: str, prompt: str) -> str:
    """Format must match nsos_curriculum_lib.format_supervised_text."""
    return f"<|task:{kind}|>\nPrompt:\n{prompt}\nAnswer:\n"


def ask(kind: str, prompt: str) -> None:
    full_prompt = build_prompt(kind, prompt)
    print(f"━━ [{kind}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Q: {prompt}")
    t = time.time()
    out = engine.generate_ex(full_prompt, opts)
    elapsed = time.time() - t
    out = out.replace("<|endoftext|>", "").strip()
    m = engine.last_generation_metrics()
    speed = m.generated_tokens / (m.elapsed_ms / 1000.0) if m.elapsed_ms > 0 else 0
    print(f"A: {out}")
    print(f"   [{m.generated_tokens} tokens, {elapsed:.1f}s, {speed:.2f} tok/s]\n")


# ── Test battery ────────────────────────────────────────────────────────
TESTS = [
    ("summarize", "Edge inference needs compact weights and predictable latency."),
    ("summarize", "BitNet uses ternary weights to reduce memory and compute."),
    ("translate", "Hello, how are you today?"),
    ("translate", "Edge inference needs compact weights."),
    ("rewrite", "this is a really bad sentence with lots of mistakes"),
    ("explain_code", "x = [i*2 for i in range(5)]"),
    ("extract_fact", "Context: NSOS is a hybrid model with Mamba and Attention. Question: What architecture does NSOS use?"),
    ("extract_fact", "Context: The cat sat on the mat. The cat is black. Question: What color is the cat?"),
]

print("═" * 60)
print(f"NSOS v10_gpu — Batch Inference Test")
print(f"Model: phase6_memory.bin (post-phase4, before polish)")
print(f"Device: CPU (PSU-safe)  |  max_tokens={opts.max_tokens}  temp={opts.temperature}")
print("═" * 60)
print()

for kind, prompt in TESTS:
    ask(kind, prompt)

print("═" * 60)
print("Done.")
