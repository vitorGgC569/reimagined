"""
Interactive chat for NSOS v10_gpu.
GPU inference via the build-cuda-validation .pyd.
Uses phase6_memory.bin (the latest checkpoint after all 4 training phases).
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
        except (AttributeError, OSError) as exc:
            print(
                f"[runtime] DLL directory registration failed for {p}: {exc}",
                file=sys.stderr,
            )

import nsos_ext as nsos  # type: ignore  # noqa: E402

# Load model on GPU
config = nsos.ModelConfig()
if MODEL_CFG.exists():
    for k, v in json.loads(MODEL_CFG.read_text("utf-8")).items():
        if hasattr(config, k):
            setattr(config, k, v)
config.use_cuda = True   # ← GPU

print(f"[load] {MODEL_BIN.name}  vocab={config.vocab_size}  layers={config.num_layers}  d_model={config.d_model}")
print(f"[load] device=GPU (CUDA)")

engine = nsos.InferenceEngine()
t0 = time.time()
if not engine.load_model(str(MODEL_BIN), config):
    print(f"[ERROR] load failed", file=sys.stderr)
    sys.exit(1)
print(f"[load] loaded in {time.time()-t0:.1f}s\n")

# Generation options — tuneable on the fly with /set
opts = nsos.GenerationOptions()
opts.max_context_tokens = 512
opts.max_tokens         = 48
opts.temperature        = 0.4
opts.top_p              = 0.85
opts.top_k              = 20
opts.stream             = False

TASK_KINDS = {
    "1": "summarize",
    "2": "extract_fact",
    "3": "explain_code",
    "4": "rewrite",
    "5": "translate",
}
TASK_HELP = {
    "summarize":    "summarize free text",
    "extract_fact": "answer from context (Context: ... Question: ...)",
    "explain_code": "explain code snippet",
    "rewrite":      "rewrite/improve text",
    "translate":    "translate text",
}


def build_prompt(kind: str, user_text: str) -> str:
    return f"<|task:{kind}|>\nPrompt:\n{user_text}\nAnswer:\n"


def print_menu():
    print()
    print("── Task types " + "─" * 40)
    for k, name in TASK_KINDS.items():
        print(f"  [{k}] {name:<13} — {TASK_HELP[name]}")
    print("  /set max_tokens=N  temp=F  top_p=F  top_k=N    (tune live)")
    print("  /info                                          (show current settings)")
    print("  /quit                                          (exit)")
    print("─" * 53)


def show_info():
    print(f"[info] task=<current> max_tokens={opts.max_tokens} temp={opts.temperature}"
          f" top_p={opts.top_p} top_k={opts.top_k}")


def parse_set(args: str) -> None:
    """Handle /set max_tokens=N temp=0.5 etc."""
    for token in args.split():
        if "=" not in token:
            print(f"[set] ignored: {token}")
            continue
        key, val = token.split("=", 1)
        try:
            if key == "max_tokens":
                opts.max_tokens = int(val)
            elif key in ("temp", "temperature"):
                opts.temperature = float(val)
            elif key == "top_p":
                opts.top_p = float(val)
            elif key == "top_k":
                opts.top_k = int(val)
            else:
                print(f"[set] unknown key: {key}")
                continue
            print(f"[set] {key} -> {val}")
        except ValueError as e:
            print(f"[set] bad value for {key}: {e}")


def main():
    print("═" * 53)
    print("  NSOS v10_gpu — Interactive Chat (GPU)")
    print("  HONEST EXPECTATIONS:")
    print("  - Model saw ~1.2M tokens (vs 750M-1.3B for Chinchilla)")
    print("  - Output will be english WORDS but lacks coherence.")
    print("  - This is a proof-of-life, not a real LLM.")
    print("═" * 53)

    current_kind = "summarize"

    while True:
        print_menu()
        sys.stdout.write(f"task=[{current_kind}] > ")
        sys.stdout.flush()
        try:
            choice = input().strip()
        except EOFError:
            break

        if not choice:
            continue
        low = choice.lower()
        if low in ("/q", "/quit", "/exit"):
            print("bye.")
            break
        if low == "/info":
            show_info()
            continue
        if low.startswith("/set"):
            parse_set(choice[4:].strip())
            continue
        if choice in TASK_KINDS:
            current_kind = TASK_KINDS[choice]
            print(f"[task] switched to: {current_kind}")
            continue

        # otherwise treat as the prompt itself
        prompt = build_prompt(current_kind, choice)

        print(f"[gen] task={current_kind} max_tokens={opts.max_tokens} temp={opts.temperature} ...")
        t = time.time()
        out = engine.generate_ex(prompt, opts)
        elapsed = time.time() - t
        out = out.replace("<|endoftext|>", "").strip()
        m = engine.last_generation_metrics()
        speed = m.generated_tokens / (m.elapsed_ms / 1000.0) if m.elapsed_ms > 0 else 0
        print(f"\nAnswer: {out}")
        print(f"[metrics] tokens={m.generated_tokens}  elapsed={elapsed:.1f}s  speed={speed:.2f} tok/s\n")


if __name__ == "__main__":
    main()
