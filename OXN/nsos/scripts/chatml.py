"""ChatML format helpers (item #19 of VISION roadmap).

ChatML is the OpenAI / Mistral / Phi convention for instruction-tuned
models.  Each turn is wrapped:

    <|im_start|>role
    content<|im_end|>

A full conversation looks like:

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    What is 2+2?<|im_end|>
    <|im_start|>assistant
    4<|im_end|>

For NSOS we register these markers as special tokens so the BPE
tokenizer keeps them as single units rather than slicing them across
many sub-tokens.  This module provides:

  * `CHATML_SPECIAL_TOKENS` — the canonical list to pass to
    `tokenizer.add_special_tokens(...)` in C++.
  * `format_chat(messages, ...)` — build a ChatML string from a list
    of {"role": "...", "content": "..."}.
  * `split_prompt_answer(messages)` — for SFT, return (prompt_text,
    answer_text) where prompt ends with the assistant header and
    answer is everything the assistant said up to the closing tag.
  * `add_function_call_tokens(...)` — bonus: helper to register
    function-call tokens used by tool-using models.

Used by:
  * sft_phase.py with `--chatml` flag
  * dpo_phase.py / safety_phase.py
  * build_sft_bundle.py to reformat raw instruction data
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


# ── Special tokens registered in the tokenizer ────────────────────────
# Order matters: tokenizer.add_special_tokens preserves insertion order,
# and downstream code (e.g. `Tokenizer::encode("<|im_start|>")[0]`) relies
# on these being a single id each.
CHATML_SPECIAL_TOKENS: List[str] = [
    "<|im_start|>",
    "<|im_end|>",
    # Tool / function-call extension (used by Mistral-Instruct, Hermes,
    # NousHermes, and increasingly the industry default).  These are
    # registered even when no tool calls are present so a future
    # function-using build doesn't require a tokenizer rebuild.
    "<|tool_call|>",
    "<|tool_response|>",
    "<|function_call|>",
    "<|function_response|>",
]

# Roles ChatML knows about.  We use bare strings (not special tokens)
# for roles — that matches OpenAI's spec and avoids vocab inflation.
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"
VALID_ROLES = (ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL)


@dataclass
class Message:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(
                f"unknown role {self.role!r}; expected one of {VALID_ROLES}"
            )


def format_chat(
    messages: Sequence[dict | Message],
    *,
    add_generation_prompt: bool = False,
    system_message: Optional[str] = None,
) -> str:
    """Render a list of messages in ChatML format.

    Args:
      messages: list of {"role": "...", "content": "..."} dicts (or Message).
      add_generation_prompt: if True, append `<|im_start|>assistant\\n`
        WITHOUT a closing tag — the right format to pass to a model for
        generating its next reply.  If False, leaves the conversation
        as-is (useful for training data where the answer is already
        included).
      system_message: optional system prompt to prepend.  Ignored if the
        first message in `messages` already has role='system'.

    Returns:
      A single string in ChatML format.

    Examples:
      >>> format_chat([{"role": "user", "content": "Hi"}], add_generation_prompt=True)
      '<|im_start|>user\\nHi<|im_end|>\\n<|im_start|>assistant\\n'

      >>> format_chat([
      ...     {"role": "user", "content": "Q"},
      ...     {"role": "assistant", "content": "A"},
      ... ])
      '<|im_start|>user\\nQ<|im_end|>\\n<|im_start|>assistant\\nA<|im_end|>\\n'
    """
    norm = [Message(**m) if isinstance(m, dict) else m for m in messages]

    parts: List[str] = []
    if system_message and (not norm or norm[0].role != ROLE_SYSTEM):
        parts.append(f"<|im_start|>system\n{system_message}<|im_end|>\n")

    for m in norm:
        parts.append(f"<|im_start|>{m.role}\n{m.content}<|im_end|>\n")

    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")

    return "".join(parts)


def split_prompt_answer(
    messages: Sequence[dict | Message],
    *,
    system_message: Optional[str] = None,
) -> Tuple[str, str]:
    """For SFT: split a conversation into (prompt, answer).

    The convention: every assistant message except the LAST one is part
    of the prompt (with full ChatML wrapping).  The last assistant
    message is the ANSWER — its content (without the closing tag) is
    returned separately so the SFT loss masks only those tokens.

    Why no closing tag in the answer: the trainer scores the answer
    tokens up to but not including <|im_end|>.  Inference will emit
    <|im_end|> naturally and we use it as a stop sequence.  Including
    <|im_end|> in the training answer would make the model learn to
    emit it even when not appropriate.

    Returns: (prompt_text, answer_text) where:
      prompt_text ends with `<|im_start|>assistant\\n`
      answer_text is the raw content of the final assistant turn.

    Raises ValueError if no assistant message present.
    """
    norm = [Message(**m) if isinstance(m, dict) else m for m in messages]
    last_assistant_idx = -1
    for i in range(len(norm) - 1, -1, -1):
        if norm[i].role == ROLE_ASSISTANT:
            last_assistant_idx = i
            break
    if last_assistant_idx < 0:
        raise ValueError("no assistant message in conversation; "
                         "cannot split for SFT")

    prompt_msgs = norm[:last_assistant_idx]
    prompt_text = format_chat(
        prompt_msgs,
        add_generation_prompt=True,
        system_message=system_message,
    )
    answer_text = norm[last_assistant_idx].content
    return prompt_text, answer_text


def register_special_tokens(tokenizer) -> None:
    """Register ChatML + function-call tokens on an `nsos_ext.Tokenizer`.

    Idempotent if the tokenizer already has them (NSOS's Tokenizer
    silently no-ops on duplicates per the C++ implementation).
    """
    tokenizer.add_special_tokens(CHATML_SPECIAL_TOKENS)


def detect_chatml(tokenizer) -> bool:
    """Check whether a tokenizer already has the ChatML markers.

    We just try encoding `<|im_start|>` and verify it tokenizes to a
    single id — if so, the special token is registered.
    """
    try:
        ids = tokenizer.encode("<|im_start|>")
    except Exception:
        return False
    return len(ids) == 1


# ── Tests / smoke ────────────────────────────────────────────────────

def _smoke():
    """Quick self-test.  Run `python -m chatml` to verify."""
    msgs = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    full = format_chat(msgs)
    assert "<|im_start|>system\nBe helpful.<|im_end|>" in full, full
    assert "<|im_start|>assistant\n4<|im_end|>" in full, full

    prompt, answer = split_prompt_answer(msgs)
    assert prompt.endswith("<|im_start|>assistant\n"), prompt
    assert answer == "4", answer

    # With generation prompt
    gen = format_chat(msgs[:-1], add_generation_prompt=True)
    assert gen.endswith("<|im_start|>assistant\n"), gen

    # System fallback
    inj = format_chat([{"role": "user", "content": "Hi"}],
                       system_message="Be terse.")
    assert "Be terse." in inj
    assert inj.index("Be terse.") < inj.index("Hi")

    print("chatml smoke OK")


if __name__ == "__main__":
    _smoke()
