"""
Interactive inference for NSOS v8 model.
Uses the correct task-format that matches the training curriculum.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

# ── UTF-8 safe stdout for Windows ────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Build dir & model paths ──────────────────────────────────────────────────
WORKTREE   = Path(__file__).resolve().parent.parent.parent.parent  # clever-roentgen-ba007c
MAIN_REPO  = Path("C:/Users/Oxta/Desktop/reimagined-main")
BUILD_DIR  = MAIN_REPO / "OXN/nsos/build-mvp/Release"
SCRIPTS    = WORKTREE / "OXN/nsos/scripts"
MODEL_DIR  = SCRIPTS / "live_distill_v8"
MODEL_BIN  = MODEL_DIR / "final_model.bin"
MODEL_CFG  = MODEL_DIR / "effective_model_config.json"

# ── Load nsos_ext ─────────────────────────────────────────────────────────────
if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))

# Windows: add DLL search paths if needed
if os.name == "nt":
    try:
        os.add_dll_directory(str(BUILD_DIR))
    except (AttributeError, OSError):
        pass

import nsos_ext as nsos  # type: ignore  # noqa: E402

# ── Load model config ─────────────────────────────────────────────────────────
config = nsos.ModelConfig()
if MODEL_CFG.exists():
    payload = json.loads(MODEL_CFG.read_text(encoding="utf-8"))
    for k, v in payload.items():
        if hasattr(config, k):
            setattr(config, k, v)
config.use_cuda = False

engine = nsos.InferenceEngine()
if not engine.load_model(str(MODEL_BIN), config):
    print(f"[ERROR] Failed to load model: {MODEL_BIN}", file=sys.stderr)
    sys.exit(1)

print(f"[OK] Model loaded: {MODEL_BIN.name}")
print(f"[OK] Vocab size: {config.vocab_size}  Layers: {config.num_layers}  d_model: {config.d_model}")
print()

# ── Generation options ────────────────────────────────────────────────────────
opts = nsos.GenerationOptions()
opts.max_context_tokens = 512
opts.max_tokens         = 64
opts.temperature        = 0.4
opts.top_p              = 0.85
opts.top_k              = 20
opts.stream             = False

# ── Task format (matches training curriculum exactly) ─────────────────────────
TASK_KINDS = {
    "1": "summarize",
    "2": "extract_fact",
    "3": "explain_code",
    "4": "rewrite",
    "5": "translate",
}

TASK_HELP = {
    "summarize":    "Provide text → model summarizes it",
    "extract_fact": "Provide context + question → model answers from context",
    "explain_code": "Paste code snippet → model explains it",
    "rewrite":      "Provide text → model rewrites/improves it",
    "translate":    "Provide text → model translates it",
}


def build_prompt(kind: str, user_text: str) -> str:
    """Format matches OXN/nsos/scripts/nsos_curriculum_lib.py:format_supervised_text"""
    return (
        f"<|task:{kind}|>\n"
        f"Prompt:\n{user_text}\n"
        f"Answer:\n"
    )


def print_menu():
    print("\n── Task types ─────────────────────────────────────")
    for k, name in TASK_KINDS.items():
        print(f"  [{k}] {name:<15} — {TASK_HELP[name]}")
    print("  [q] quit")
    print("────────────────────────────────────────────────────")


def main():
    print("════════════════════════════════════════════════════")
    print("  NSOS v8 — Interactive Inference")
    print("  ~0.12 tok/s on CPU  |  64 tokens ≈ 9 min")
    print("  Use short prompts for faster responses!")
    print("════════════════════════════════════════════════════")

    current_kind = "summarize"

    while True:
        print_menu()
        choice = input(f"Select task [{current_kind}]> ").strip()

        if choice.lower() in ("q", "quit", "exit", "/exit"):
            print("Bye!")
            break

        if choice in TASK_KINDS:
            current_kind = TASK_KINDS[choice]
            print(f"[task] Switched to: {current_kind}")

        print(f"\n[{current_kind}] Enter prompt (empty line = done):")
        lines = []
        while True:
            try:
                line = input("  > ")
            except EOFError:
                break
            if line == "" and lines:
                break
            lines.append(line)

        if not lines:
            continue

        user_text = "\n".join(lines)
        prompt = build_prompt(current_kind, user_text)

        print(f"\n[generating] task={current_kind}, max_tokens={opts.max_tokens}, temp={opts.temperature}")
        print("─" * 52)

        output = engine.generate_ex(prompt, opts)
        # Strip only the endoftext token and leading/trailing whitespace
        output = output.replace("<|endoftext|>", "").strip()

        print(f"Answer: {output}")

        metrics = engine.last_generation_metrics()
        tok_s = metrics.generated_tokens / (metrics.elapsed_ms / 1000.0) if metrics.elapsed_ms > 0 else 0
        print(f"\n[metrics] tokens={metrics.generated_tokens}  "
              f"time={metrics.elapsed_ms/1000:.1f}s  "
              f"speed={tok_s:.2f} tok/s")
        print("─" * 52)


if __name__ == "__main__":
    main()
