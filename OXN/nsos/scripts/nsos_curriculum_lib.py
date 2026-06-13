from __future__ import annotations

import hashlib
import json
import os
import random
import re
import struct
import textwrap
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|sep|>"]

PHASE_ORDER = [
    "phase1_algorithms",
    "phase2_structured",
    "phase3_curated_text",
    "phase4_instructions",
    "phase5_verifier",
    "phase6_memory",
]

# LEARN A2 (Curriculum reorder — Arxiv 2405.07490, 2506.11300, 2601.21698):
# v11 curriculum order changes the EXECUTION sequence from the historical
# "algorithms → structured → text → instructions → verifier → memory" to
# "curated_text → algorithms → structured → instructions → verifier → memory".
#
# Rationale (curriculum learning research 2024-2026):
#   * Curated text (phase 3 in v11: TinyStories + Cosmopedia + Wikipedia
#     + C4) is the EASIEST distribution for an under-trained model to
#     fit — high lexical regularity, natural narrative flow, large
#     sample count.  Starting with it builds the basic language model.
#   * Algorithms (phase 1) is the HARDEST distribution at this scale
#     because outputs are exact short strings ("EVEN", "11010", "12")
#     with no fluency safety net.  Asking a from-scratch model to
#     learn that first poisons the first-token distribution.
#   * Structured QA (phase 2) needs the model to already know how
#     English sentences work — it learns "answer is a span of the
#     evidence", which assumes it can read.
#   * Instructions (phase 4) is composition of all the above.
#   * Verifier (5) and Memory (6) are specialty phases at the end.
#
# Empirically (Strategic Data Ordering, Arxiv 2405.07490): curriculum
# learning gives a notable boost in early-to-mid training and is most
# useful when the model has limited capacity (our 40M case).
#
# The names of the phases are kept (so existing checkpoints and
# bundle layouts continue to load) but the execution order changes.
# train_curriculum.py reads PHASE_ORDER_V11 when the profile sets
# `curriculum_phase_order = "v11"`, else falls back to PHASE_ORDER.
PHASE_ORDER_V11 = [
    "phase3_curated_text",   # easiest distribution → foundation
    "phase1_algorithms",     # symbolic math after basic English
    "phase2_structured",     # extractive QA on top of comprehension
    "phase4_instructions",   # compose: read + answer briefly
    "phase5_verifier",       # specialty: verify previous answers
    "phase6_memory",         # specialty: multi-turn memory
]


def resolve_phase_order(profile_curriculum_order: str | None = None) -> list:
    """Return the phase execution order for a given profile setting.

    profile_curriculum_order:
        - None or "v10": use the historical PHASE_ORDER
          (algorithms → structured → text → instructions → verifier → memory)
        - "v11": use PHASE_ORDER_V11 (curated_text first; curriculum-learning
          easier-to-harder ordering per Arxiv 2405.07490)
    """
    if profile_curriculum_order == "v11":
        return list(PHASE_ORDER_V11)
    return list(PHASE_ORDER)

DEFAULT_PHASE_SIZES = {
    "phase1_algorithms": {"train": 512, "eval": 128},
    "phase2_structured": {"train": 384, "eval": 96},
    "phase3_curated_text": {"train": 1536, "eval": 192},
    "phase4_instructions": {"train": 512, "eval": 128},
    "phase5_verifier": {"train": 384, "eval": 96},
    "phase6_memory": {"train": 512, "eval": 128},
}

# v11 preset: scales up phase budgets so the model sees materially more
# unique data per epoch.  Sizing rationale (assumes Cosmopedia + TinyStories
# + SmolTalk are present in artifacts/real_datasets/):
#
#   phase1 algorithms: 4× — still small, but enough to absorb scaled
#                      orca_math + deepmind_math + svamp pulls
#   phase2 structured: 5× — adds SQuAD v2 as a third evidence source
#                      next to CoQA; needs more rows to cover SQuAD's variety
#   phase3 curated_text: 25× — THE headline change.  The 30,000-row
#                      target gives ~15M tokens from a Cosmopedia/Wikipedia/
#                      TinyStories/C4/Stack mix, which is enough that the
#                      v10 plateau at loss 7.3 should actually move.
#   phase4 instructions: 8× — adds SmolTalk (4,000 modern synthetic
#                      conversations) to the xlsum + opus_books mix
#   phase5 verifier: 4× — proportional bump
#   phase6 memory: 5× — proportional bump; CoQA has the rows to support it
DEFAULT_PHASE_SIZES_V11 = {
    "phase1_algorithms":   {"train":  2000, "eval": 512},
    "phase2_structured":   {"train":  2000, "eval": 512},
    "phase3_curated_text": {"train": 30000, "eval": 1000},
    "phase4_instructions": {"train":  4000, "eval": 1024},
    "phase5_verifier":     {"train":  1500, "eval": 384},
    "phase6_memory":       {"train":  2500, "eval": 640},
}

COUNT_LABELS = ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN"]
COUNT_LABELS_PT = ["ZERO", "UM", "DOIS", "TRES", "QUATRO", "CINCO", "SEIS", "SETE"]
TOKENIZER_PHASE_TEXT_CAPS = {
    "phase1_algorithms": 160,
    "phase2_structured": 160,
    "phase3_curated_text": 1200,
    "phase4_instructions": 320,
    "phase5_verifier": 160,
    "phase6_memory": 240,
}
TOKENIZER_PHASE_CHAR_BUDGETS = {
    "phase1_algorithms": 28_000,
    "phase2_structured": 28_000,
    "phase3_curated_text": 320_000,
    "phase4_instructions": 88_000,
    "phase5_verifier": 28_000,
    "phase6_memory": 52_000,
}

REAL_DATASET_FILES = {
    "orca_math_word_problems": "orca_math_word_problems.jsonl",
    "deepmind_math_large": "deepmind_math_large.jsonl",
    "svamp": "svamp.jsonl",
    "triviaqa_rc_wikipedia": "triviaqa_rc_wikipedia.jsonl",
    "coqa": "coqa.jsonl",
    "wikitext_en": "wikitext_en.jsonl",
    "wikipedia_en": "wikipedia_en.jsonl",
    "wikipedia_pt": "wikipedia_pt.jsonl",
    "xlsum_en": "xlsum_en.jsonl",
    "xlsum_pt": "xlsum_pt.jsonl",
    "opus_books_en_pt": "opus_books_en_pt.jsonl",
    # v11 expansion (added 2026-05-16)
    "cosmopedia_v2": "cosmopedia_v2.jsonl",
    "tinystories": "tinystories.jsonl",
    "smoltalk": "smoltalk.jsonl",
    "the_stack_smol": "the_stack_smol.jsonl",
    "c4_sample": "c4_sample.jsonl",
    "squad_v2": "squad_v2.jsonl",
    # LEARN A1: all eight Cosmopedia configs (Cosmopedia paper trained
    # cosmo-1B on the full mix totaling ~25B tokens).  Each config
    # targets a different register; we sample from all to keep phase 3
    # text registry diverse.
    "cosmopedia_khanacademy": "cosmopedia_khanacademy.jsonl",
    "cosmopedia_openstax":    "cosmopedia_openstax.jsonl",
    "cosmopedia_stanford":    "cosmopedia_stanford.jsonl",
    "cosmopedia_stories":     "cosmopedia_stories.jsonl",
    "cosmopedia_web_v1":      "cosmopedia_web_v1.jsonl",
    "cosmopedia_web_v2":      "cosmopedia_web_v2.jsonl",
    "cosmopedia_wikihow":     "cosmopedia_wikihow.jsonl",
}
_REAL_DATASET_CACHE: Dict[str, List[Dict]] = {}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def format_supervised_text(kind: str, prompt: str, answer: str) -> str:
    return (
        f"<|task:{kind}|>\n"
        f"Prompt:\n{prompt}\n"
        f"Answer:\n{answer}<|endoftext|>"
    )


def format_document_text(title: str, body: str, source: str) -> str:
    return (
        f"<|document|>\n"
        f"Title: {title}\n"
        f"Source: {source}\n\n"
        f"{body.strip()}\n"
        f"<|endoftext|>"
    )


def chunk_text(text: str, chunk_chars: int = 720, overlap_chars: int = 96) -> List[str]:
    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not normalized:
        return []
    if len(normalized) <= chunk_chars:
        return [normalized]

    chunks: List[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_chars, len(normalized))
        if end < len(normalized):
            split = normalized.rfind("\n", start, end)
            if split <= start + chunk_chars // 2:
                split = normalized.rfind(" ", start, end)
            if split > start:
                end = split
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def _make_record(phase: str, kind: str, prompt: str, answer: str, source: str = "synthetic") -> Dict:
    text = format_supervised_text(kind, prompt, answer)
    return {
        "id": stable_hash(f"{phase}:{kind}:{prompt}:{answer}"),
        "phase": phase,
        "kind": kind,
        "prompt": prompt,
        "answer": answer,
        "source": source,
        "text": text,
    }


def _make_doc_record(phase: str, title: str, body: str, source: str) -> Dict:
    text = format_document_text(title, body, source)
    return {
        "id": stable_hash(f"{phase}:{title}:{source}:{body[:128]}"),
        "phase": phase,
        "kind": "document",
        "prompt": "",
        "answer": "",
        "source": source,
        "text": text,
    }


def _random_token_string(rng: random.Random, min_len: int = 6, max_len: int = 18) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    length = rng.randint(min_len, max_len)
    return "".join(rng.choice(alphabet) for _ in range(length))


def _boolean_gate(op: str, a: int, b: int) -> int:
    if op == "AND":
        return a & b
    if op == "OR":
        return a | b
    if op == "XOR":
        return a ^ b
    if op == "NAND":
        return 1 - (a & b)
    if op == "NOR":
        return 1 - (a | b)
    if op == "XNOR":
        return 1 - (a ^ b)
    raise ValueError(f"Unknown gate: {op}")


def _pick_split_subset(
    items: Sequence[Tuple[str, str, str]],
    split: str,
    rng: random.Random,
    train_fraction: float = 0.8,
) -> List[Tuple[str, str, str]]:
    if not items:
        return []
    shuffled = list(items)
    rng.shuffle(shuffled)
    if len(shuffled) == 1:
        return shuffled
    cut = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * train_fraction))))
    chosen = shuffled[:cut] if split == "train" else shuffled[cut:]
    return chosen or shuffled


def _real_dataset_dir(repo_root: Path) -> Path:
    return repo_root / "OXN" / "nsos" / "artifacts" / "real_datasets"


def _load_real_dataset_rows(repo_root: Path, name: str) -> List[Dict]:
    filename = REAL_DATASET_FILES.get(name)
    if filename is None:
        return []
    path = _real_dataset_dir(repo_root) / filename
    cache_key = str(path)
    cached = _REAL_DATASET_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not path.exists():
        _REAL_DATASET_CACHE[cache_key] = []
        return []
    rows = read_jsonl(path)
    _REAL_DATASET_CACHE[cache_key] = rows
    return rows


def _pick_split_rows(
    rows: Sequence[Dict],
    split: str,
    rng: random.Random,
    train_fraction: float = 0.82,
    max_rows: int | None = None,
) -> List[Dict]:
    if not rows:
        return []
    shuffled = list(rows)
    rng.shuffle(shuffled)
    if len(shuffled) == 1:
        chosen = shuffled
    else:
        cut = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * train_fraction))))
        chosen = shuffled[:cut] if split == "train" else shuffled[cut:]
        if not chosen:
            chosen = shuffled
    if max_rows is not None:
        return chosen[:max_rows]
    return chosen


def _normalize_inline_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _expand_records(records: List[Dict], count: int, rng: random.Random) -> List[Dict]:
    if len(records) >= count:
        rng.shuffle(records)
        return records[:count]
    expanded = list(records)
    cursor = 0
    while records and len(expanded) < count:
        template = records[cursor % len(records)]
        duplicate = dict(template)
        duplicate["id"] = stable_hash(f"{template['id']}:{len(expanded)}")
        expanded.append(duplicate)
        cursor += 1
    rng.shuffle(expanded)
    return expanded[:count]


def build_phase1_algorithms(count: int, seed: int, split: str) -> List[Dict]:
    rng = random.Random(seed)
    rows: List[Dict] = []
    weighted_modes = (
        ["copy_short"] * 3
        + ["reverse_short"] * 3
        + ["binary_add"] * 2
        + ["compare_label"] * 2
        + ["parity_label"] * 1
        + ["count_label"] * 1
    )

    pt = os.environ.get("NSOS_CURRICULUM_LANG", "").lower() == "pt"
    for _ in range(count):
        mode = rng.choice(weighted_modes)
        if mode == "copy_short":
            value = _random_token_string(rng, 4, 8)
            prompt = (f"Copie exatamente esta sequencia: {value}" if pt
                      else f"Copy exactly this token stream: {value}")
            answer = value
        elif mode == "reverse_short":
            value = _random_token_string(rng, 4, 8)
            prompt = (f"Inverta esta sequencia: {value}" if pt
                      else f"Reverse this token stream: {value}")
            answer = value[::-1]
        elif mode == "parity_label":
            bits = "".join(rng.choice("01") for _ in range(rng.randint(5, 9)))
            even = sum(bit == "1" for bit in bits) % 2 == 0
            if pt:
                prompt = f"Paridade de {bits}. Responda PAR ou IMPAR."
                answer = "PAR" if even else "IMPAR"
            else:
                prompt = f"Parity for {bits}. Answer with EVEN or ODD."
                answer = "EVEN" if even else "ODD"
        elif mode == "binary_add":
            a = rng.randint(0, 15)
            b = rng.randint(0, 15)
            prompt = (f"Some os valores binarios {a:b} + {b:b}. Responda somente em binario."
                      if pt else
                      f"Add the binary values {a:b} + {b:b}. Answer in binary only.")
            answer = format(a + b, "b")
        elif mode == "compare_label":
            a = rng.randint(-20, 20)
            b = rng.randint(-20, 20)
            if pt:
                prompt = f"Compare {a} e {b}. Responda com um rotulo: MENOR, MAIOR ou IGUAL."
                answer = "MENOR" if a < b else "MAIOR" if a > b else "IGUAL"
            else:
                prompt = f"Compare {a} and {b}. Answer with one label from LT, GT, EQ."
                answer = "LT" if a < b else "GT" if a > b else "EQ"
        else:
            target = rng.choice("abcxyz012")
            base = _random_token_string(rng, 6, 10)
            mixed = base + target * rng.randint(0, 4)
            chars = list(mixed)
            rng.shuffle(chars)
            shuffled = "".join(chars)
            count_value = min(shuffled.count(target), len(COUNT_LABELS) - 1)
            if pt:
                prompt = (f"Conte quantas vezes '{target}' aparece em: {shuffled}. "
                          "Responda com um rotulo de ZERO a SETE.")
                answer = COUNT_LABELS_PT[count_value]
            else:
                prompt = (
                    f"Count how many times '{target}' appears in: {shuffled}. "
                    "Answer with one label from ZERO to SEVEN."
                )
                answer = COUNT_LABELS[count_value]

        rows.append(_make_record("phase1_algorithms", mode, prompt, answer, f"{split}_synthetic"))
    return rows


def build_phase2_structured(count: int, seed: int, split: str) -> List[Dict]:
    rng = random.Random(seed)
    rows: List[Dict] = []
    actors = ["nina", "paulo", "lia", "omar", "maya", "yuri", "sofia", "ravi"]
    actions = ["compile", "deploy", "rotate", "resume", "audit", "stage", "mirror", "shard"]
    states = ["ready", "stale", "fresh", "quiet", "rapid", "steady", "dense", "clear"]
    zones = ["oslo", "porto", "recife", "kyoto", "lima", "tunis", "siena", "helsinki"]
    config_values = {
        "profile": ["cedar", "nova", "quill", "ember", "tidal", "opal", "spruce", "harbor"],
        "backend": ["raster", "vector", "binary", "mixer", "triton", "spiral", "anchor", "fable"],
        "region": ["north", "south", "east", "west", "delta", "canal", "ridge", "basin"],
        "tier": ["core", "pilot", "edge", "batch", "prime", "swift", "calm", "dual"],
        "codec": ["flint", "prism", "linen", "woven", "cinder", "petal", "glyph", "coral"],
    }

    for _ in range(count):
        mode = rng.choice(["circuit", "dsl", "log_extract", "config_lookup"])
        if mode == "circuit":
            op = rng.choice(["AND", "OR", "XOR", "NAND", "NOR", "XNOR"])
            a = rng.randint(0, 1)
            b = rng.randint(0, 1)
            prompt = f"Circuit solve: gate={op} A={a} B={b}. Answer with 0 or 1."
            answer = str(_boolean_gate(op, a, b))
        elif mode == "dsl":
            value = rng.randint(1, 7)
            ops: List[str] = [f"SET {value}"]
            current = value
            for _step in range(rng.randint(2, 4)):
                cmd = rng.choice(["ADD", "MUL", "SUB"])
                amount = rng.randint(1, 4)
                ops.append(f"{cmd} {amount}")
                if cmd == "ADD":
                    current += amount
                elif cmd == "MUL":
                    current *= amount
                else:
                    current -= amount
            prompt = "Execute this mini DSL and answer with the final integer: " + " | ".join(ops)
            answer = str(current)
        elif mode == "log_extract":
            line = (
                f"[{rng.choice(['INFO', 'WARN', 'TRACE'])}] "
                f"actor={rng.choice(actors)} "
                f"action={rng.choice(actions)} "
                f"state={rng.choice(states)} "
                f"zone={rng.choice(zones)}"
            )
            field = rng.choice(["actor", "action", "state", "zone"])
            prompt = f"Read this structured log line and return the {field} value only:\n{line}"
            answer = re.search(rf"{field}=([a-z]+)", line).group(1)
        else:
            settings = {key: rng.choice(values) for key, values in config_values.items()}
            ask_key = rng.choice(list(settings.keys()))
            prompt = textwrap.dedent(
                f"""
                Read this configuration block and return the {ask_key} value only.
                profile={settings['profile']}
                backend={settings['backend']}
                region={settings['region']}
                tier={settings['tier']}
                codec={settings['codec']}
                """
            ).strip()
            answer = settings[ask_key]

        rows.append(_make_record("phase2_structured", mode, prompt, answer, f"{split}_synthetic"))
    return rows


def _story_window(text: str, answer_start: int, radius: int = 280) -> str:
    normalized = text.strip()
    if not normalized:
        return ""
    if answer_start < 0:
        return _normalize_inline_text(normalized[: radius * 2])
    start = max(answer_start - radius, 0)
    end = min(answer_start + radius, len(normalized))
    if start > 0:
        split = normalized.rfind(" ", 0, start)
        if split > 0:
            start = split + 1
    if end < len(normalized):
        split = normalized.find(" ", end)
        if split > 0:
            end = split
    return _normalize_inline_text(normalized[start:end])


def _phase1_real_rows(repo_root: Path, split: str, seed: int) -> List[Dict]:
    rows: List[Dict] = []
    specs = [
        ("orca_math_word_problems", 840 if split == "train" else 192, "math_word_problem"),
        ("deepmind_math_large", 640 if split == "train" else 144, "math_exact"),
        ("svamp", 220 if split == "train" else 60, "math_word_problem"),
    ]
    for offset, (dataset_name, max_rows, kind) in enumerate(specs):
        dataset_rows = _pick_split_rows(
            _load_real_dataset_rows(repo_root, dataset_name),
            split,
            random.Random(seed + 71 + offset * 13),
            max_rows=max_rows,
        )
        for row in dataset_rows:
            answer = _normalize_inline_text(str(row.get("answer", "")))
            if not answer:
                continue
            if dataset_name in {"deepmind_math_large", "orca_math_word_problems"}:
                question = _normalize_inline_text(row.get("question", ""))
                if len(question) < 12:
                    continue
                prompt = f"Solve exactly and answer with the final result only.\nProblem:\n{question}"
            else:
                body = _normalize_inline_text(row.get("body", ""))
                question = _normalize_inline_text(row.get("question", ""))
                if len(body) < 12 or len(question) < 6:
                    continue
                prompt = (
                    "Read the word problem and answer with the final number only.\n"
                    f"Problem:\n{body} {question}"
                )
            rows.append(_make_record("phase1_algorithms", kind, prompt, answer, row.get("source", dataset_name)))
    return rows


def build_phase1_algorithms_v2(repo_root: Path, count: int, seed: int, split: str) -> List[Dict]:
    rng = random.Random(seed)
    rows = ([] if os.environ.get("NSOS_CURRICULUM_LANG", "").lower() == "pt"
            else list(_phase1_real_rows(repo_root, split, seed)))
    rng.shuffle(rows)
    real_target = min(len(rows), max(count // 2, int(count * 0.7)))
    mixed = rows[:real_target]
    mixed.extend(build_phase1_algorithms(max(count - len(mixed), 0), seed + 401, split))
    return _expand_records(mixed, count, rng)


def _phase2_real_rows(repo_root: Path, split: str, seed: int) -> List[Dict]:
    """Source phase-2 structured rows (evidence_extract).

    v10 baseline: CoQA only (multi-turn QA over stories, ~360 train rows).
    v11 expansion: ADDITIONALLY sources SQuAD v2 — short factual answers
    from Wikipedia passages.  SQuAD complements CoQA because:
      * CoQA answers are conversation-style ("oh, the cat"); SQuAD
        answers are clean spans ("the cat")
      * CoQA contexts are stories; SQuAD contexts are factual paragraphs
      * Together they teach the model to extract precise spans across
        registers (narrative + encyclopedic)

    SQuAD v2 also contains unanswerable questions which filter_squad_v2
    drops, so every row we get here has a verifiable answer.
    """
    rows: List[Dict] = []

    # ── CoQA (multi-turn, conversational extraction) ───────────────────
    coqa_rows = _pick_split_rows(
        _load_real_dataset_rows(repo_root, "coqa"),
        split,
        random.Random(seed + 173),
        max_rows=800 if split == "train" else 200,
    )
    for row in coqa_rows:
        story = row.get("story", "")
        qa_pairs = row.get("questions", []) or []
        for pair in qa_pairs[:5]:
            question = _normalize_inline_text(pair.get("question", ""))
            answer = _normalize_inline_text(pair.get("answer", ""))
            answer_start = int(pair.get("answer_start", -1))
            context = _story_window(story, answer_start)
            if len(question) < 8 or len(answer) < 1 or len(context) < 80:
                continue
            prompt = textwrap.dedent(
                f"""
                Use the evidence snippet and return the shortest exact answer span only.
                Evidence: {context}
                Question: {question}
                """
            ).strip()
            rows.append(_make_record("phase2_structured", "evidence_extract", prompt, answer, row.get("source", "coqa")))

    # ── v11: SQuAD v2 (Wikipedia factual extraction) ───────────────────
    squad_rows = _pick_split_rows(
        _load_real_dataset_rows(repo_root, "squad_v2"),
        split,
        random.Random(seed + 197),
        max_rows=1500 if split == "train" else 400,
    )
    for row in squad_rows:
        question = _normalize_inline_text(row.get("question", ""))
        context = _normalize_inline_text(row.get("context", ""))
        answer = _normalize_inline_text(row.get("answer", ""))
        if len(question) < 8 or len(context) < 50 or not answer:
            continue
        # Truncate long context to keep within seq_len; preserve the part
        # around the answer if possible (simple find).
        if len(context) > 800:
            answer_pos = context.find(answer)
            if answer_pos >= 0:
                start = max(answer_pos - 350, 0)
                end = min(start + 800, len(context))
                context = context[start:end]
            else:
                context = context[:800]
        prompt = textwrap.dedent(
            f"""
            Use the evidence snippet and return the shortest exact answer span only.
            Evidence: {context}
            Question: {question}
            """
        ).strip()
        rows.append(_make_record("phase2_structured", "evidence_extract", prompt, answer,
                                  row.get("source", "rajpurkar/squad_v2")))

    return rows


def build_phase2_structured_v2(repo_root: Path, count: int, seed: int, split: str) -> List[Dict]:
    rng = random.Random(seed)
    rows = list(_phase2_real_rows(repo_root, split, seed))
    rng.shuffle(rows)
    real_target = min(len(rows), max(count // 2, int(count * 0.65)))
    mixed = rows[:real_target]
    mixed.extend(build_phase2_structured(max(count - len(mixed), 0), seed + 433, split))
    return _expand_records(mixed, count, rng)


def _phase6_real_rows(repo_root: Path, split: str, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    rows: List[Dict] = []
    dataset_rows = _pick_split_rows(
        _load_real_dataset_rows(repo_root, "coqa"),
        split,
        random.Random(seed + 281),
        max_rows=320 if split == "train" else 72,
    )
    for row in dataset_rows:
        story = row.get("story", "")
        qa_pairs = row.get("questions", []) or []
        if len(story) < 120 or len(qa_pairs) < 2:
            continue
        for pair_index, pair in enumerate(qa_pairs[:6]):
            question = _normalize_inline_text(pair.get("question", ""))
            answer = _normalize_inline_text(pair.get("answer", ""))
            answer_start = int(pair.get("answer_start", -1))
            if len(question) < 4 or not answer:
                continue
            snippet = _story_window(story, answer_start)
            if len(snippet) < 80:
                continue
            if pair_index > 0 and rng.random() < 0.6:
                history = qa_pairs[max(0, pair_index - 2):pair_index]
                prompt_lines = [
                    "Conversation memory task. Use the story excerpt and the previous turns.",
                    f"Story excerpt: {snippet}",
                ]
                for turn in history:
                    prev_q = _normalize_inline_text(turn.get("question", ""))
                    prev_a = _normalize_inline_text(turn.get("answer", ""))
                    if prev_q and prev_a:
                        prompt_lines.append(f"User: {prev_q}")
                        prompt_lines.append(f"Assistant: {prev_a}")
                prompt_lines.append(f"User: {question}")
                prompt_lines.append("Assistant:")
                prompt = "\n".join(prompt_lines)
                kind = "conversation_recall"
            else:
                prompt = textwrap.dedent(
                    f"""
                    Read the story excerpt, keep the details in memory, and answer with the shortest exact span only.
                    Story excerpt: {snippet}
                    Question: {question}
                    Answer:
                    """
                ).strip()
                kind = "story_recall"
            rows.append(_make_record("phase6_memory", kind, prompt, answer, row.get("source", "coqa")))
    return rows


def build_phase6_memory_v3(repo_root: Path, count: int, seed: int, split: str) -> List[Dict]:
    rng = random.Random(seed)
    rows = list(_phase6_real_rows(repo_root, split, seed))
    rng.shuffle(rows)
    real_target = min(len(rows), max(count // 2, int(count * 0.7)))
    mixed = rows[:real_target]
    mixed.extend(build_phase6_memory_v2(max(count - len(mixed), 0), seed + 467, split))
    return _expand_records(mixed, count, rng)


def _repo_documents(repo_root: Path) -> List[Tuple[str, Path]]:
    candidates = [
        ("NSOS Root README", repo_root / "OXN" / "nsos" / "README.md"),
        ("NSOS LLM Plan", repo_root / "OXN" / "nsos" / "docs" / "NSOS_LLM_SMALL_PLAN.md"),
        ("NSOS Validation Status", repo_root / "OXN" / "nsos" / "docs" / "NSOS_VALIDATION_STATUS.md"),
        ("OxtaMem NSOS Notes", repo_root / "modules" / "oxtamem" / "README_NSOS.md"),
    ]
    return [(title, path) for title, path in candidates if path.exists()]


def _repo_code_documents(repo_root: Path) -> List[Tuple[str, Path]]:
    candidates = [
        ("NSOS Jamba Header", repo_root / "OXN" / "nsos" / "include" / "jamba.h"),
        ("NSOS Jamba Source", repo_root / "OXN" / "nsos" / "src" / "jamba.cpp"),
        ("NSOS Mamba2 Source", repo_root / "OXN" / "nsos" / "src" / "mamba2.cpp"),
        ("NSOS BitLinear Source", repo_root / "OXN" / "nsos" / "src" / "bitlinear.cpp"),
        ("NSOS Trainer Source", repo_root / "OXN" / "nsos" / "src" / "trainer.cpp"),
        ("NSOS Tensor Source", repo_root / "OXN" / "nsos" / "src" / "tensor.cpp"),
    ]
    return [(title, path) for title, path in candidates if path.exists()]


def _phase3_real_documents(repo_root: Path, split: str, seed: int) -> List[Tuple[str, str, str]]:
    """Source phase-3 documents from real datasets.

    v10 baseline: Wikipedia EN/PT + WikiText only.
    v11 expansion: ADDITIONALLY sources from Cosmopedia (synthetic
    educational, designed for small LMs), TinyStories (simple narrative,
    proves coherence in small models per Eldridge & Li 2023), C4 sample
    (cleaned web text), and The Stack (Python code, code-aware chunking).

    Each new dataset is gated on file existence — if the user hasn't
    fetched it via fetch_real_datasets.py, _load_real_dataset_rows()
    returns [] and that source is silently skipped.  This means the same
    nsos_curriculum_lib.py works for v10 (no new datasets fetched) and
    v11 (new datasets present) without code branching.

    Per-source row limits scale with the v11 phase budget — when the
    caller wants 30,000 phase-3 rows (DEFAULT_PHASE_SIZES_V11), we pull
    more aggressively from the larger sources.  The split parameter
    'train' vs 'eval' is honored by _pick_split_rows; each source uses
    a different RNG offset to avoid correlation."""
    # NSOS_CURRICULUM_LANG=pt: modo PT-first — o produto fala portugues; corta
    # ingles e codigo da fase de texto (o raio-x G6 + o teste qualitativo
    # mostraram a "moda codigo": pool unico ~6:1 codigo:texto real).
    if os.environ.get("NSOS_CURRICULUM_LANG", "").lower() == "pt":
        dataset_specs = [
            ("wikipedia_pt", 620, 72, 24000, 1200),
        ]
        docs: List[Tuple[str, str, str]] = []
        for offset, (dataset_name, chunk_chars, overlap_chars,
                     max_rows_train, max_rows_eval) in enumerate(dataset_specs):
            max_rows = max_rows_train if split == "train" else max_rows_eval
            rows = _pick_split_rows(
                _load_real_dataset_rows(repo_root, dataset_name),
                split,
                random.Random(seed + 101 + offset * 17),
                max_rows=max_rows,
            )
            for row_index, row in enumerate(rows):
                text = str(row.get("text", "") or "")
                for ci, chunk in enumerate(
                        chunk_text(text, chunk_chars=chunk_chars,
                                   overlap_chars=overlap_chars)):
                    docs.append((f"{dataset_name} r{row_index} #{ci + 1}",
                                 chunk, dataset_name))
        return docs

    dataset_specs = [
        # (dataset_name, chunk_chars, overlap_chars, max_rows_train, max_rows_eval)
        # ── v10 baseline (always pulled) ─────────────────────────────────
        ("wikipedia_en", 620, 72, 4000, 240),
        ("wikipedia_pt", 620, 72, 4000, 240),
        ("wikitext_en",  560, 64, 2000, 120),
        # ── v11 expansion (skipped if not fetched) ───────────────────────
        # Cosmopedia (LEARN A1 — full multi-config sweep, 2026-05-16):
        # We pull from all 8 configs to diversify the synthetic register.
        # cosmopedia_v2 is the auto_math_text config kept for backward
        # compat with the v11 first-pass build that only fetched it.
        # Longer chunks (720) because passages are coherent multi-paragraph.
        ("cosmopedia_v2",          720, 80, 4000, 200),
        ("cosmopedia_khanacademy", 720, 80, 4000, 200),
        ("cosmopedia_openstax",    720, 80, 4000, 200),
        ("cosmopedia_stanford",    720, 80, 4000, 200),
        ("cosmopedia_stories",     560, 64, 4000, 200),  # narrative — shorter chunks
        ("cosmopedia_web_v1",      720, 80, 4000, 200),
        ("cosmopedia_web_v2",      720, 80, 4000, 200),
        ("cosmopedia_wikihow",     560, 64, 4000, 200),  # how-to — shorter chunks
        # TinyStories: simple narrative; small chunks (320) because each
        # story is short on purpose.  Eldridge & Li (2023) shows these
        # alone make 10-30M models coherent.
        ("tinystories",   320, 32, 4000, 200),
        # C4: cleaned web text; mid-length chunks for sentence variety.
        ("c4_sample",     580, 64, 2500, 160),
        # The Stack (Python only): code documents.  Smaller chunks (420)
        # because per-token information density is higher than prose.
        ("the_stack_smol", 420, 56, 1500, 120),
    ]
    docs: List[Tuple[str, str, str]] = []
    for offset, (dataset_name, chunk_chars, overlap_chars,
                 max_rows_train, max_rows_eval) in enumerate(dataset_specs):
        max_rows = max_rows_train if split == "train" else max_rows_eval
        rows = _pick_split_rows(
            _load_real_dataset_rows(repo_root, dataset_name),
            split,
            random.Random(seed + 101 + offset * 17),
            max_rows=max_rows,
        )
        if not rows:
            # Dataset wasn't fetched (or fetch failed) — skip silently.
            # This is the gate that lets v10 callers keep working.
            continue
        for row in rows:
            # Each dataset stores its document text under slightly different
            # keys; the filters in fetch_real_datasets.py normalize most
            # of them to `text`, but `the_stack_smol` uses `content`.
            if dataset_name == "the_stack_smol":
                raw = row.get("content", "").strip()
                title = row.get("path") or "stack_python_file"
            else:
                raw = row.get("text", "").strip()
                title = row.get("title") or row.get("id") or dataset_name
            if not raw:
                continue
            source = row.get("source", dataset_name)
            for index, chunk in enumerate(
                chunk_text(raw, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
            ):
                docs.append((f"{title} #{index + 1}", chunk, source))
    return docs


def _phase4_real_rows(repo_root: Path, split: str, seed: int) -> List[Dict]:
    """Source phase-4 instruction rows.

    v10 baseline: xlsum_en + xlsum_pt (summarization) + opus_books_en_pt
                  (translation).  Total ~400 train rows max.
    v11 expansion: ADDITIONALLY sources SmolTalk (HuggingFaceTB) for
                  modern synthetic instructions — replaces aging Alpaca-
                  style data with conversation-format prompts that are
                  better matched to small-model context windows.

    SmolTalk filter (in fetch_real_datasets.py::filter_smoltalk) already
    flattens multi-turn conversations into single (prompt, answer)
    pairs taking the first user→assistant exchange.  We classify these
    under the kind "instruction_conversation" so the trainer can balance
    them against the structured tasks.
    """
    rows: List[Dict] = []

    # ── xlsum (summarization) ──────────────────────────────────────────
    for offset, dataset_name in enumerate(("xlsum_en", "xlsum_pt")):
        dataset_rows = _pick_split_rows(
            _load_real_dataset_rows(repo_root, dataset_name),
            split,
            random.Random(seed + 211 + offset * 19),
            max_rows=800 if split == "train" else 200,
        )
        for row in dataset_rows:
            text = _normalize_inline_text(row.get("text", ""))
            summary = _normalize_inline_text(row.get("summary", ""))
            title = _normalize_inline_text(row.get("title", ""))
            if len(text) < 260 or len(summary) < 24:
                continue
            source = row.get("source", dataset_name)
            if row.get("lang") == "pt":
                prompt = f"Resuma em uma frase curta em português.\nTítulo: {title}\nTexto:\n{text[:960]}"
            else:
                prompt = f"Write one short summary sentence in English.\nTitle: {title}\nText:\n{text[:960]}"
            rows.append(_make_record("phase4_instructions", "summarize", prompt, summary, source))

    # ── opus_books (translation) ───────────────────────────────────────
    opus_rows = _pick_split_rows(
        _load_real_dataset_rows(repo_root, "opus_books_en_pt"),
        split,
        random.Random(seed + 263),
        max_rows=1000 if split == "train" else 240,
    )
    for row in opus_rows:
        english = _normalize_inline_text(row.get("en", ""))
        portuguese = _normalize_inline_text(row.get("pt", ""))
        if len(english) < 16 or len(portuguese) < 16:
            continue
        if len(english.split()) <= 2 or len(portuguese.split()) <= 2:
            continue
        source = row.get("source", "Helsinki-NLP/opus_books:en-pt")
        rows.append(
            _make_record(
                "phase4_instructions",
                "translate_pt",
                f"Translate to Portuguese:\n{english}",
                portuguese,
                source,
            )
        )
        rows.append(
            _make_record(
                "phase4_instructions",
                "translate_en",
                f"Translate to English:\n{portuguese}",
                english,
                source,
            )
        )

    # ── v11: SmolTalk (modern synthetic instructions) ──────────────────
    smoltalk_rows = _pick_split_rows(
        _load_real_dataset_rows(repo_root, "smoltalk"),
        split,
        random.Random(seed + 311),
        max_rows=3000 if split == "train" else 600,
    )
    for row in smoltalk_rows:
        prompt = _normalize_inline_text(row.get("prompt", ""))
        answer = _normalize_inline_text(row.get("answer", ""))
        if len(prompt) < 8 or len(answer) < 8:
            continue
        # Cap prompt to keep within model seq_len budget; answer is already
        # capped at 800 chars by filter_smoltalk.
        if len(prompt) > 600:
            prompt = prompt[:600].rsplit(" ", 1)[0] + "…"
        source = row.get("source", "HuggingFaceTB/smoltalk")
        rows.append(_make_record("phase4_instructions",
                                  "instruction_conversation",
                                  prompt, answer, source))

    return rows


def _phase4_synthetic_rows(count: int, seed: int, split: str) -> List[Dict]:
    rng = random.Random(seed)
    rows: List[Dict] = []
    systems = ["runtime", "decoder", "trainer", "memory layer", "checkpoint loader", "tokenizer", "scheduler", "edge pack"]
    verbs = ["stabilizes", "protects", "preserves", "reduces", "simplifies", "aligns", "keeps", "improves"]
    qualities = ["clean prompts", "exact labels", "compact weights", "stable checkpoints", "shared cache", "predictable latency"]
    metrics = ["loss", "latency", "memory use", "token throughput", "checkpoint quality", "prompt fidelity"]
    modules = ["value", "state", "buffer", "logit", "cache", "route"]
    for index in range(count):
        mode = rng.choice(["rewrite", "explain_code", "extract_fact", "summarize"])
        if mode == "rewrite":
            system = rng.choice(systems)
            quality = rng.choice(qualities)
            prompt = (
                "Rewrite this sentence to sound cleaner and more technical:\n"
                f"We need the {system} to stay focused on {quality} while still being easy to debug."
            )
            answer = f"The {system} should stay focused on {quality} while remaining easy to debug."
        elif mode == "explain_code":
            module = rng.choice(modules)
            scale = rng.randint(2, 5)
            shift = rng.randint(1, 4)
            code = (
                f"{module} = {module} * {scale}\n"
                f"{module} = {module} - {shift}\n"
                f"return {module}"
            )
            answer = f"The code multiplies {module} by {scale}, subtracts {shift}, and returns the result."
            prompt = f"Explain in one sentence what this code does:\n{code}"
        elif mode == "extract_fact":
            system = rng.choice(systems)
            metric = rng.choice(metrics)
            verb = rng.choice(verbs)
            quality = rng.choice(qualities)
            passage = (
                f"The {system} {verb} {metric} during long runs. "
                f"It also preserves {quality} so the checkpoints remain easier to compare."
            )
            question = rng.choice(
                [
                    f"What does the {system} improve?",
                    f"What does the {system} preserve?",
                ]
            )
            answer = metric if "improve" in question else quality
            prompt = f"Read the passage and answer briefly.\nPassage:\n{passage}\nQuestion: {question}"
        else:
            system = rng.choice(systems)
            verb = rng.choice(verbs)
            quality = rng.choice(qualities)
            prompt = (
                "Write one short summary sentence for this text:\n"
                f"The {system} {verb} {quality} during long runs and keeps the pipeline easier to audit."
            )
            answer = f"The {system} {verb} {quality} during long runs."

        rows.append(_make_record("phase4_instructions", mode, prompt, answer, f"{split}_synthetic"))
    return rows


def _phase3_reference_documents(split: str) -> List[Tuple[str, str, str]]:
    train_docs = [
        (
            "Compact language modeling",
            (
                "A compact language model improves fastest when it first learns clean sentence structure, "
                "stable vocabulary usage, and short factual continuations. Technical PT and EN text are useful "
                "because the syntax is regular and the concepts repeat across files."
            ),
            "handwritten",
        ),
        (
            "Exact answers matter",
            (
                "The NSOS runtime should prefer short exact answers when a task is verifiable. A good small model "
                "must know when to answer with one token, when to emit a short label, and when to continue a document."
            ),
            "handwritten",
        ),
        (
            "Code and systems text",
            (
                "Code comments, API notes, and systems documentation teach the model how to move between natural "
                "language and structured syntax. This improves instruction following for edge tooling, memory, and "
                "debugging tasks."
            ),
            "handwritten",
        ),
        (
            "Bilingual technical writing",
            (
                "O modelo deve entender linguagem tecnica em portugues e em ingles. Mixed bilingual corpora help a "
                "small network keep names, commands, and code terms aligned across both languages."
            ),
            "handwritten",
        ),
        (
            "Instruction style",
            (
                "Short technical instructions should map cleanly to short exact outputs. A compact model benefits "
                "from examples that alternate between explanation, translation, formatting, and factual extraction."
            ),
            "handwritten",
        ),
        (
            "Memory and retrieval",
            (
                "Session memory tasks work best when the model first learns stable lexical patterns. Retrieval quality "
                "depends on language competence, not only on a memory module."
            ),
            "handwritten",
        ),
        (
            "Edge engineering notes",
            (
                "GPU utilization, packed inference, and checkpoint selection are engineering topics that should appear "
                "as prose, not only as raw code. Small models learn better from repeated explanations of these ideas."
            ),
            "handwritten",
        ),
        (
            "Replay discipline",
            (
                "Replay is useful only when it preserves task identity. If every family of task is replayed into every "
                "other family, the model learns a blurred distribution and starts with the wrong token."
            ),
            "handwritten",
        ),
        (
            "Tokenizer pressure",
            (
                "A tokenizer that overfits short structured outputs will merge them into dominant tokens. Weighting "
                "the tokenizer toward natural text helps preserve sentence level continuation and reduces degenerate starts."
            ),
            "handwritten",
        ),
        (
            "Champion selection",
            (
                "The best checkpoint for one phase is not always the best checkpoint overall. A reliable small model "
                "needs a global champion selected across multiple held out suites."
            ),
            "handwritten",
        ),
    ]
    eval_docs = [
        (
            "Held out technical note",
            (
                "Inference quality is not the same as training loss. A stable model must continue technical text, "
                "follow formatting instructions, and preserve exact keys during transformations."
            ),
            "handwritten_eval",
        ),
        (
            "Held out bilingual note",
            (
                "Sessao de memoria, resposta curta, e texto tecnico limpo precisam coexistir. The best checkpoint is "
                "the one that preserves language quality while still keeping exact structured answers."
            ),
            "handwritten_eval",
        ),
        (
            "Held out curriculum note",
            (
                "Text pretraining should dominate before instruction tuning. Structured supervision is useful later, "
                "but it should not poison the first token distribution of free generation."
            ),
            "handwritten_eval",
        ),
    ]
    return train_docs if split == "train" else eval_docs


def build_phase3_curated_text(repo_root: Path, count: int, seed: int, split: str) -> List[Dict]:
    return build_phase3_curated_text_v2(repo_root, count, seed, split)


def build_phase4_instructions(count: int, seed: int, split: str) -> List[Dict]:
    rng = random.Random(seed)
    rows: List[Dict] = []

    systems = ["runtime", "decoder", "trainer", "memory layer", "checkpoint loader", "tokenizer", "scheduler", "edge pack"]
    verbs = ["improves", "reduces", "preserves", "stabilizes", "simplifies", "aligns", "organizes", "protects"]
    qualities = [
        "predictable latency",
        "compact weights",
        "clean prompts",
        "exact labels",
        "stable checkpoints",
        "bilingual docs",
        "shared cache",
        "safe fallbacks",
    ]
    translate_pairs = [
        ("The runtime keeps exact labels for verifier tasks.", "O runtime preserva rotulos exatos para tarefas de verificacao."),
        ("A compact model still needs clean technical prose.", "Um modelo compacto ainda precisa de prosa tecnica limpa."),
        ("The tokenizer should favor natural text over noisy patterns.", "O tokenizador deve favorecer texto natural em vez de padroes ruidosos."),
        ("Replay must respect task families to avoid contamination.", "O replay precisa respeitar familias de tarefa para evitar contaminacao."),
        ("Session memory improves when the prompts stay unambiguous.", "A memoria de sessao melhora quando os prompts permanecem sem ambiguidade."),
        ("A global champion checkpoint is more reliable than a phase local winner.", "Um checkpoint campeao global e mais confiavel do que um vencedor local de phase."),
    ]
    reverse_translate_pairs = [
        ("O decoder incremental reaproveita o prefixo compartilhado.", "The incremental decoder reuses the shared prefix."),
        ("A fase textual deve dominar antes da instrucao supervisionada.", "The textual phase should dominate before supervised instruction."),
        ("Respostas muito curtas criam um prior ruim para o primeiro token.", "Very short answers create a bad prior for the first token."),
        ("A avaliacao precisa misturar loss e geracao real.", "Evaluation needs to mix loss and real generation."),
    ]
    code_pairs = [
        (
            "if (token == eos) break; output.push_back(token);",
            "The code stops at eos and otherwise appends the token to the output.",
        ),
        (
            "value = value.rmsnorm(); logits = proj.forward(value); return logits;",
            "The snippet normalizes the value, projects it, and returns the logits.",
        ),
        (
            "for (int i = 0; i < n; ++i) acc += values[i]; return acc;",
            "The loop sums the values and returns the accumulated result.",
        ),
        (
            "if (score > best_score) best_score = score; save_checkpoint();",
            "The code keeps the best score and saves a checkpoint when it improves.",
        ),
    ]
    fact_passages = [
        (
            "The edge runtime writes tokenizer artifacts before training starts. It saves the tokenizer pack inside the run directory for reuse.",
            "Where does the runtime save the tokenizer pack?",
            "inside the run directory",
        ),
        (
            "The verifier expects short exact labels, while the text phase expects fluent continuations. Mixing those signals too early harms generation.",
            "What does the verifier expect?",
            "short exact labels",
        ),
        (
            "A global champion compares checkpoints across multiple held out suites. This prevents one strong local phase from dominating the final model.",
            "Why is a global champion useful?",
            "it prevents one local phase from dominating the final model",
        ),
        (
            "Tokenizer weighting favors phase3 prose so that sentence level continuation remains healthy. Structured answers are still learned, but they stop dominating merges.",
            "What does tokenizer weighting favor?",
            "phase3 prose",
        ),
    ]

    for _ in range(count):
        mode = rng.choice(["summarize", "rewrite", "translate_pt", "translate_en", "explain_code", "extract_fact"])
        if mode == "summarize":
            system = rng.choice(systems)
            verb = rng.choice(verbs)
            quality = rng.choice(qualities)
            source = (
                f"The {system} {verb} {quality} during long runs. "
                "It also keeps the pipeline auditable for compact technical models."
            )
            prompt = f"Write one short summary sentence for this text:\n{source}"
            answer = f"The {system} {verb} {quality} during long runs."
        elif mode == "rewrite":
            system = rng.choice(systems)
            quality = rng.choice(qualities)
            prompt = (
                "Rewrite this sentence to sound cleaner and more technical:\n"
                f"We really want the {system} to stay super focused on {quality} without getting messy."
            )
            answer = f"We want the {system} to stay focused on {quality} without becoming messy."
        elif mode == "translate_pt":
            source, answer = rng.choice(translate_pairs)
            prompt = f"Translate to Portuguese:\n{source}"
        elif mode == "translate_en":
            source, answer = rng.choice(reverse_translate_pairs)
            prompt = f"Translate to English:\n{source}"
        elif mode == "explain_code":
            code, answer = rng.choice(code_pairs)
            prompt = f"Explain in one sentence what this code does:\n{code}"
        else:
            passage, question, answer = rng.choice(fact_passages)
            prompt = f"Read the passage and answer briefly.\nPassage:\n{passage}\nQuestion: {question}"

        rows.append(_make_record("phase4_instructions", mode, prompt, answer, f"{split}_synthetic"))
    return rows

    summary_pairs = [
        (
            "NSOS trains on structured tasks before broad language data. This improves stability and exactness.",
            "NSOS starts with structured tasks to improve stability and exactness.",
        ),
        (
            "Packed ternary inference reduces memory traffic, but it only works well when export and runtime agree on layout.",
            "Packed ternary inference needs matching export and runtime layouts.",
        ),
        (
            "A small bilingual technical model should preserve commands, identifiers, and key names across Portuguese and English.",
            "A bilingual technical model should preserve commands and key names across PT and EN.",
        ),
    ]
    json_pairs = [
        {"name": "nsos", "mode": "api", "lang": "pt", "tier": "small"},
        {"name": "oxtamem", "mode": "edge", "lang": "en", "tier": "core"},
        {"name": "bitnet", "mode": "train", "lang": "en", "tier": "qat"},
    ]
    classify_pairs = [
        ("error: cuda synchronize failed after kernel launch", "error"),
        ("warning: using fallback tokenizer bundle", "warning"),
        ("status: all tests passed and checkpoint saved", "status"),
    ]
    rewrite_pairs = [
        (
            "The runtime is very, very fast and kind of stable but maybe still a bit rough in places.",
            "The runtime is fast and fairly stable, but it still needs polish.",
        ),
        (
            "We want a response that is short, exact, and easy to verify.",
            "Respond briefly, exactly, and in a verifiable format.",
        ),
    ]
    translate_pairs = [
        ("The session stores facts and recalls them later.", "A sessao guarda fatos e os recupera depois."),
        ("Edge inference needs compact weights and predictable latency.", "Inferencia em edge precisa de pesos compactos e latencia previsivel."),
        ("The verifier should answer with a short exact label.", "O verificador deve responder com um rotulo curto e exato."),
    ]
    explain_pairs = [
        (
            "int acc = 0; for (int i = 0; i < n; ++i) acc += values[i]; return acc;",
            "This loop sums all values and returns the accumulated total.",
        ),
        (
            "if (token == eos) break; output.push_back(token);",
            "The code stops at eos and otherwise appends the token to the output.",
        ),
        (
            "x = x.rmsnorm(); y = proj.forward(x); return y.relu();",
            "The snippet normalizes the input, projects it, and applies ReLU.",
        ),
    ]

    for _ in range(count):
        mode = rng.choice(
            ["summarize", "convert_json", "classify", "rewrite", "translate", "explain_code"]
        )
        if mode == "summarize":
            source, answer = rng.choice(summary_pairs)
            prompt = f"Write one short summary sentence for this text:\n{source}"
        elif mode == "convert_json":
            payload = dict(rng.choice(json_pairs))
            prompt = (
                "Convert these key=value pairs to compact JSON with keys ordered as "
                "name,mode,lang,tier: "
                f"name={payload['name']} mode={payload['mode']} lang={payload['lang']} tier={payload['tier']}"
            )
            answer = json.dumps(payload, separators=(",", ":"))
        elif mode == "classify":
            text, answer = rng.choice(classify_pairs)
            prompt = (
                "Classify the log line with one label from {status,warning,error}:\n"
                f"{text}"
            )
        elif mode == "rewrite":
            source, answer = rng.choice(rewrite_pairs)
            prompt = f"Rewrite this sentence to be cleaner and more technical:\n{source}"
        elif mode == "translate":
            source, answer = rng.choice(translate_pairs)
            prompt = f"Translate to Portuguese:\n{source}"
        else:
            code, answer = rng.choice(explain_pairs)
            prompt = f"Explain in one sentence what this code does:\n{code}"

        rows.append(_make_record("phase4_instructions", mode, prompt, answer, f"{split}_synthetic"))
    return rows

    templates = [
        (
            "summarize",
            "Resuma em uma frase curta: {text}",
            lambda value: f"Resumo: {value.split('.')[0].strip()}.",
        ),
        (
            "convert_json",
            "Converta para JSON compacto: name={name} mode={mode} lang={lang}",
            lambda value: json.dumps(value, separators=(",", ":")),
        ),
        (
            "explain_code",
            "Explique em uma frase o que este trecho faz:\n{code}",
            lambda value: value,
        ),
        (
            "translate",
            "Translate to Portuguese: {text}",
            lambda value: value,
        ),
    ]

    explanation_bank = [
        "This loop accumulates a running sum and returns the final value.",
        "Este trecho aplica uma multiplicacao simples e devolve o resultado.",
        "The function normalizes input values before the projection step.",
    ]
    translate_bank = [
        ("The model stores session facts and recalls them later.", "O modelo armazena fatos da sessao e os recupera depois."),
        ("Edge inference needs compact weights and predictable latency.", "Inferencia em edge precisa de pesos compactos e latencia previsivel."),
    ]
    summarize_bank = [
        "NSOS trains on structured tasks before broad language data. This improves stability and exactness.",
        "BitNet style inference reduces memory usage with packed ternary weights. It still needs a careful export path.",
    ]

    for _ in range(count):
        mode, prompt_template, answer_builder = rng.choice(templates)
        if mode == "summarize":
            value = rng.choice(summarize_bank)
            prompt = prompt_template.format(text=value)
            answer = answer_builder(value)
        elif mode == "convert_json":
            value = {
                "name": rng.choice(["nsos", "oxtamem", "bitnet"]),
                "mode": rng.choice(["edge", "train", "api"]),
                "lang": rng.choice(["pt", "en"]),
            }
            prompt = prompt_template.format(**value)
            answer = answer_builder(value)
        elif mode == "explain_code":
            code = rng.choice(
                [
                    "int acc = 0; for (int i = 0; i < n; ++i) acc += values[i]; return acc;",
                    "x = x.rmsnorm(); y = proj.forward(x); return y.relu();",
                    "if (token == eos) break; output.push_back(token);",
                ]
            )
            prompt = prompt_template.format(code=code)
            answer = rng.choice(explanation_bank)
        else:
            src, dst = rng.choice(translate_bank)
            prompt = prompt_template.format(text=src)
            answer = dst

        rows.append(_make_record("phase4_instructions", mode, prompt, answer, f"{split}_synthetic"))
    return rows


def build_phase5_verifier(count: int, seed: int, split: str) -> List[Dict]:
    rng = random.Random(seed)
    rows: List[Dict] = []

    for _ in range(count):
        mode = rng.choice(["math_small", "compare_label", "boolean_gate", "parity_label", "code_output"])
        if mode == "math_small":
            a = rng.randint(0, 24)
            b = rng.randint(0, 24)
            op = rng.choice(["+", "-"])
            prompt = f"Compute exactly and answer with one integer: {a} {op} {b}"
            answer = str(a + b if op == "+" else a - b)
        elif mode == "compare_label":
            a = rng.randint(-24, 24)
            b = rng.randint(-24, 24)
            prompt = f"Compare {a} and {b}. Answer with one label from LT, GT, EQ."
            answer = "LT" if a < b else "GT" if a > b else "EQ"
        elif mode == "boolean_gate":
            op = rng.choice(["AND", "OR", "XOR"])
            a = rng.randint(0, 1)
            b = rng.randint(0, 1)
            prompt = f"Evaluate {op}({a},{b}). Answer with TRUE or FALSE."
            answer = "TRUE" if _boolean_gate(op, a, b) else "FALSE"
        elif mode == "parity_label":
            bits = "".join(rng.choice("01") for _ in range(rng.randint(4, 10)))
            prompt = f"Parity for {bits}. Answer with EVEN or ODD."
            answer = "EVEN" if sum(bit == "1" for bit in bits) % 2 == 0 else "ODD"
        else:
            x = rng.randint(1, 6)
            y = rng.randint(1, 6)
            z = rng.randint(0, 4)
            prompt = (
                "What does this Python snippet print? Answer with one integer.\n"
                f"v = {x}\n"
                f"v = v + {y}\n"
                f"v = v - {z}\n"
                "print(v)"
            )
            answer = str(x + y - z)

        rows.append(_make_record("phase5_verifier", mode, prompt, answer, f"{split}_synthetic"))
    return rows

    for _ in range(count):
        mode = rng.choice(["math_small", "compare", "boolean_gate", "parity", "code_output"])
        if mode == "math_small":
            a = rng.randint(0, 24)
            b = rng.randint(0, 24)
            op = rng.choice(["+", "-"])
            prompt = f"Compute exactly and answer with one integer: {a} {op} {b}"
            answer = str(a + b if op == "+" else a - b)
        elif mode == "compare":
            a = rng.randint(-20, 20)
            b = rng.randint(-20, 20)
            prompt = f"Compare {a} and {b}. Answer with one token from <, >, =."
            answer = "<" if a < b else ">" if a > b else "="
        elif mode == "boolean_gate":
            op = rng.choice(["AND", "OR", "XOR"])
            a = rng.randint(0, 1)
            b = rng.randint(0, 1)
            prompt = f"Evaluate {op}({a},{b}). Answer with 0 or 1."
            answer = str(_boolean_gate(op, a, b))
        elif mode == "parity":
            bits = "".join(rng.choice("01") for _ in range(rng.randint(4, 10)))
            prompt = f"Parity for {bits}. Answer 0 for even ones and 1 for odd ones."
            answer = str(sum(bit == "1" for bit in bits) % 2)
        else:
            x = rng.randint(1, 5)
            y = rng.randint(1, 5)
            prompt = (
                "What does this Python snippet print? Answer with one integer.\n"
                f"v = {x}\n"
                f"v = v + {y}\n"
                "print(v)"
            )
            answer = str(x + y)

        rows.append(_make_record("phase5_verifier", mode, prompt, answer, f"{split}_synthetic"))
    return rows

    for _ in range(count):
        mode = rng.choice(["math", "boolean_formula", "code_output"])
        if mode == "math":
            a = rng.randint(10, 99)
            b = rng.randint(10, 99)
            op = rng.choice(["+", "-", "*"])
            prompt = f"Compute exactly: {a} {op} {b}"
            answer = str(eval(f"{a}{op}{b}"))
        elif mode == "boolean_formula":
            a = rng.randint(0, 1)
            b = rng.randint(0, 1)
            c = rng.randint(0, 1)
            prompt = f"Evaluate (({a} XOR {b}) AND {c}). Answer with 0 or 1."
            answer = str((a ^ b) & c)
        else:
            x = rng.randint(1, 9)
            y = rng.randint(1, 9)
            z = rng.randint(1, 9)
            prompt = (
                "What is the output of this Python snippet?\n"
                f"print(({x} + {y}) * {z})"
            )
            answer = str((x + y) * z)

        rows.append(_make_record("phase5_verifier", mode, prompt, answer, f"{split}_synthetic"))
    return rows


def build_phase6_memory(count: int, seed: int, split: str) -> List[Dict]:
    return build_phase6_memory_v2(count, seed, split)

    rng = random.Random(seed)
    rows: List[Dict] = []
    owners = ["selene", "orion", "maia", "nolan", "iris", "vega", "lucan", "sora", "talin", "mira"]
    badges = ["amber", "cobalt", "fennel", "ivory", "juniper", "mosaic", "onyx", "saffron", "topaz", "violet"]
    routes = ["atlas", "beacon", "cedar", "delta", "ember", "glacier", "harbor", "iona", "mistral", "solstice"]
    modules = ["lumen", "quill", "rivet", "solace", "tundra", "vortex", "willow", "zephyr", "cinder", "petal"]
    styles = ["nimble", "steady", "lucid", "quiet", "precise", "vivid", "measured", "brisk", "gentle", "stark"]

    for _ in range(count):
        facts = {
            "owner": rng.choice(owners),
            "badge": rng.choice(badges),
            "route": rng.choice(routes),
            "module": rng.choice(modules),
            "style": rng.choice(styles),
        }
        mode = rng.choice(["direct_recall", "overwrite_recall", "pair_recall", "fact_table"])

        if mode == "direct_recall":
            ask_key = rng.choice(["badge", "route", "module", "style"])
            prompt = textwrap.dedent(
                f"""
                System: remember the following session profile.
                User: owner={facts['owner']}
                Assistant: stored.
                User: badge={facts['badge']}
                Assistant: stored.
                User: route={facts['route']}
                Assistant: stored.
                User: module={facts['module']}
                Assistant: stored.
                User: style={facts['style']}
                Assistant: stored.
                User: what is the {ask_key}? Reply with the stored value only.
                Assistant:
                """
            ).strip()
            answer = facts[ask_key]
        elif mode == "overwrite_recall":
            key = rng.choice(["badge", "route", "module", "style"])
            pools = {
                "badge": badges,
                "route": routes,
                "module": modules,
                "style": styles,
            }
            replacement_pool = [value for value in pools[key] if value != facts[key]]
            updated = rng.choice(replacement_pool)
            prompt = textwrap.dedent(
                f"""
                System: keep session memory updated.
                User: {key}={facts[key]}
                Assistant: stored.
                User: update {key}={updated}
                Assistant: updated.
                User: what is the {key} now? Reply with the new value only.
                Assistant:
                """
            ).strip()
            answer = updated
        elif mode == "pair_recall":
            prompt = textwrap.dedent(
                f"""
                System: remember the owner profile.
                User: owner={facts['owner']}
                Assistant: stored.
                User: module={facts['module']}
                Assistant: stored.
                User: route={facts['route']}
                Assistant: stored.
                User: which module belongs to {facts['owner']}? Reply with one word only.
                Assistant:
                """
            ).strip()
            answer = facts["module"]
        else:
            prompt = textwrap.dedent(
                f"""
                System: remember this compact session card.
                User:
                owner:{facts['owner']}
                badge:{facts['badge']}
                route:{facts['route']}
                module:{facts['module']}
                style:{facts['style']}
                Assistant: stored.
                User: return the badge for this card. Answer with one word only.
                Assistant:
                """
            ).strip()
            answer = facts["badge"]

        rows.append(_make_record("phase6_memory", mode, prompt, answer, f"{split}_synthetic"))
    return rows

    names = ["ana", "bruno", "caio", "dora"]
    cities = ["oslo", "lima", "kyoto", "recife"]
    tools = ["edge", "train", "cache", "agent"]
    langs = ["pt", "en", "es", "de"]
    tones = ["calm", "bold", "clean", "exact"]

    for _ in range(count):
        mode = rng.choice(["direct_recall", "overwrite_recall", "pair_recall", "fact_table"])
        facts = {
            "name": rng.choice(names),
            "city": rng.choice(cities),
            "tool": rng.choice(tools),
            "lang": rng.choice(langs),
            "tone": rng.choice(tones),
        }

        if mode == "direct_recall":
            ask_key = rng.choice(["city", "tool", "lang", "tone"])
            transcript = textwrap.dedent(
                f"""
                System: remember the following facts for this session.
                User: name={facts['name']}
                Assistant: stored.
                User: city={facts['city']}
                Assistant: stored.
                User: tool={facts['tool']}
                Assistant: stored.
                User: lang={facts['lang']}
                Assistant: stored.
                User: tone={facts['tone']}
                Assistant: stored.
                User: what is the {ask_key}? Reply with the stored value only.
                Assistant:
                """
            ).strip()
            answer = facts[ask_key]
        elif mode == "overwrite_recall":
            key = rng.choice(["city", "tool", "lang", "tone"])
            original = facts[key]
            replacement_pool = [
                value
                for value in {"city": cities, "tool": tools, "lang": langs, "tone": tones}[key]
                if value != original
            ]
            updated = rng.choice(replacement_pool)
            transcript = textwrap.dedent(
                f"""
                System: keep session memory updated.
                User: {key}={original}
                Assistant: stored.
                User: update {key}={updated}
                Assistant: updated.
                User: what is the {key} now? Reply with the new value only.
                Assistant:
                """
            ).strip()
            answer = updated
        elif mode == "fact_table":
            transcript = textwrap.dedent(
                f"""
                System: remember this compact profile table.
                User:
                name:{facts['name']}
                city:{facts['city']}
                tool:{facts['tool']}
                lang:{facts['lang']}
                tone:{facts['tone']}
                Assistant: stored.
                User: return the tool for this profile. Answer with one word only.
                Assistant:
                """
            ).strip()
            answer = facts["tool"]
        else:
            transcript = textwrap.dedent(
                f"""
                System: remember the user profile.
                User: name={facts['name']}
                Assistant: stored.
                User: tool={facts['tool']}
                Assistant: stored.
                User: Which tool belongs to {facts['name']}? Answer with one word only.
                Assistant:
                """
            ).strip()
            answer = facts["tool"]

        rows.append(_make_record("phase6_memory", mode, transcript, answer, f"{split}_synthetic"))
    return rows

    for _ in range(count):
        facts = {
            "name": rng.choice(names),
            "city": rng.choice(cities),
            "tool": rng.choice(tools),
            "lang": rng.choice(langs),
        }
        ask_key = rng.choice(list(facts.keys()))
        transcript = textwrap.dedent(
            f"""
            System: store the following session facts.
            User: name={facts['name']}
            Assistant: memorized.
            User: city={facts['city']}
            Assistant: memorized.
            User: tool={facts['tool']}
            Assistant: memorized.
            User: lang={facts['lang']}
            Assistant: memorized.
            User: What is the {ask_key}?
            Assistant:
            """
        ).strip()
        rows.append(_make_record("phase6_memory", "memory_recall", transcript, facts[ask_key], f"{split}_synthetic"))
    return rows


def build_phase3_curated_text_v2(repo_root: Path, count: int, seed: int, split: str) -> List[Dict]:
    rng = random.Random(seed)
    docs: List[Dict] = []
    real_docs = _phase3_real_documents(repo_root, split, seed)
    random.Random(seed + 13).shuffle(real_docs)

    if os.environ.get("NSOS_CURRICULUM_LANG", "").lower() == "pt":
        # PT-first: so corpus real PT (sem handwritten EN, sem repo, sem codigo);
        # limite = count (era 560 — o gargalo que fazia o codigo dominar o pool).
        for title, chunk, source in real_docs[:count]:
            docs.append(_make_doc_record("phase3_curated_text", title, chunk, source))
        return _expand_records(docs, count, rng)

    for title, body, source in _phase3_reference_documents(split):
        for index, chunk in enumerate(chunk_text(body, chunk_chars=360, overlap_chars=48)):
            docs.append(_make_doc_record("phase3_curated_text", f"{title} #{index + 1}", chunk, source))

    real_limit = 560 if split == "train" else 112
    for title, chunk, source in real_docs[:real_limit]:
        docs.append(_make_doc_record("phase3_curated_text", title, chunk, source))

    prose_chunks: List[Tuple[str, str, str]] = []
    for title, path in _repo_documents(repo_root):
        raw = read_text_strict(path)
        for index, chunk in enumerate(chunk_text(raw, chunk_chars=560, overlap_chars=64)):
            prose_chunks.append((f"{title} #{index + 1}", chunk, str(path.relative_to(repo_root))))

    code_chunks: List[Tuple[str, str, str]] = []
    for title, path in _repo_code_documents(repo_root):
        raw = read_text_strict(path)
        tagged = f"File: {path.name}\n\n{raw}"
        for index, chunk in enumerate(chunk_text(tagged, chunk_chars=420, overlap_chars=56)):
            code_chunks.append((f"{title} #{index + 1}", chunk, str(path.relative_to(repo_root))))

    for title, chunk, source in _pick_split_subset(prose_chunks, split, random.Random(seed + 17)):
        docs.append(_make_doc_record("phase3_curated_text", title, chunk, source))

    code_limit = max(1, count // 8)
    for title, chunk, source in _pick_split_subset(code_chunks, split, random.Random(seed + 31))[:code_limit]:
        docs.append(_make_doc_record("phase3_curated_text", title, chunk, source))

    return _expand_records(docs, count, rng)


def build_phase4_instructions_v2(repo_root: Path, count: int, seed: int, split: str) -> List[Dict]:
    rng = random.Random(seed)
    rows = list(_phase4_real_rows(repo_root, split, seed))
    rng.shuffle(rows)
    real_target = max(count // 2, int(count * 0.7))
    rows = rows[:real_target]
    rows.extend(_phase4_synthetic_rows(max(count - len(rows), 0), seed + 401, split))
    return _expand_records(rows, count, rng)


def build_phase6_memory_v2(count: int, seed: int, split: str) -> List[Dict]:
    rng = random.Random(seed)
    rows: List[Dict] = []
    owners = ["selene", "orion", "maia", "nolan", "iris", "vega", "lucan", "sora", "talin", "mira", "althea", "dorian"]
    badges = ["amber", "cobalt", "fennel", "ivory", "juniper", "mosaic", "onyx", "saffron", "topaz", "violet", "vermilion", "marble"]
    routes = ["atlas", "beacon", "cedar", "delta", "ember", "glacier", "harbor", "iona", "mistral", "solstice", "tangent", "aurora"]
    modules = ["lumen", "quill", "rivet", "solace", "tundra", "vortex", "willow", "zephyr", "cinder", "petal", "heather", "signal"]
    styles = ["nimble", "steady", "lucid", "quiet", "precise", "vivid", "measured", "brisk", "gentle", "stark", "polished", "crisp"]
    weighted_modes = (
        ["direct_recall"] * 4
        + ["overwrite_recall"] * 3
        + ["pair_recall"] * 2
        + ["fact_table"] * 1
    )

    for _ in range(count):
        facts = {
            "owner": rng.choice(owners),
            "badge": rng.choice(badges),
            "route": rng.choice(routes),
            "module": rng.choice(modules),
            "style": rng.choice(styles),
        }
        mode = rng.choice(weighted_modes)

        if mode == "direct_recall":
            ask_key = rng.choice(["badge", "route", "module", "style"])
            prompt = textwrap.dedent(
                f"""
                System: remember the following session profile exactly.
                User: owner={facts['owner']}
                Assistant: stored.
                User: badge={facts['badge']}
                Assistant: stored.
                User: route={facts['route']}
                Assistant: stored.
                User: module={facts['module']}
                Assistant: stored.
                User: style={facts['style']}
                Assistant: stored.
                User: what is the {ask_key}? Reply with the stored one-word value only.
                Assistant:
                """
            ).strip()
            answer = facts[ask_key]
        elif mode == "overwrite_recall":
            key = rng.choice(["badge", "route", "module", "style"])
            pools = {
                "badge": badges,
                "route": routes,
                "module": modules,
                "style": styles,
            }
            updated = rng.choice([value for value in pools[key] if value != facts[key]])
            prompt = textwrap.dedent(
                f"""
                System: keep session memory updated.
                User: {key}={facts[key]}
                Assistant: stored.
                User: update {key}={updated}
                Assistant: updated.
                User: what is the {key} now? Reply with the new one-word value only.
                Assistant:
                """
            ).strip()
            answer = updated
        elif mode == "pair_recall":
            prompt = textwrap.dedent(
                f"""
                System: remember the owner profile.
                User: owner={facts['owner']}
                Assistant: stored.
                User: module={facts['module']}
                Assistant: stored.
                User: route={facts['route']}
                Assistant: stored.
                User: which module belongs to {facts['owner']}? Reply with one word only.
                Assistant:
                """
            ).strip()
            answer = facts["module"]
        else:
            prompt = textwrap.dedent(
                f"""
                System: remember this compact session card.
                User:
                owner:{facts['owner']}
                badge:{facts['badge']}
                route:{facts['route']}
                module:{facts['module']}
                style:{facts['style']}
                Assistant: stored.
                User: return the badge for this card. Answer with one word only.
                Assistant:
                """
            ).strip()
            answer = facts["badge"]

        rows.append(_make_record("phase6_memory", mode, prompt, answer, f"{split}_synthetic"))
    return rows


def build_curriculum(repo_root: Path,
                     out_dir: Path,
                     seed: int = 1337,
                     phase_sizes: Dict[str, Dict[str, int]] | None = None) -> Path:
    phase_sizes = phase_sizes or DEFAULT_PHASE_SIZES
    ensure_dir(out_dir)
    data_dir = ensure_dir(out_dir / "data")

    builders = {
        "phase1_algorithms": lambda c, s, split: build_phase1_algorithms_v2(repo_root, c, s, split),
        "phase2_structured": lambda c, s, split: build_phase2_structured_v2(repo_root, c, s, split),
        "phase3_curated_text": lambda c, s, split: build_phase3_curated_text_v2(repo_root, c, s, split),
        "phase4_instructions": lambda c, s, split: build_phase4_instructions_v2(repo_root, c, s, split),
        "phase5_verifier": lambda c, s, split: build_phase5_verifier(c, s, split),
        "phase6_memory": lambda c, s, split: build_phase6_memory_v3(repo_root, c, s, split),
    }

    manifest = {
        "seed": seed,
        "special_tokens": SPECIAL_TOKENS,
        "phases": [],
    }

    for phase_index, phase_name in enumerate(PHASE_ORDER):
        sizes = phase_sizes[phase_name]
        train_rows = builders[phase_name](sizes["train"], seed + phase_index * 1000 + 11, "train")
        eval_rows = builders[phase_name](sizes["eval"], seed + phase_index * 1000 + 29, "eval")

        train_path = data_dir / f"{phase_name}.train.jsonl"
        eval_path = data_dir / f"{phase_name}.eval.jsonl"
        write_jsonl(train_path, train_rows)
        write_jsonl(eval_path, eval_rows)

        manifest["phases"].append(
            {
                "name": phase_name,
                "train_file": str(train_path.relative_to(out_dir)),
                "eval_file": str(eval_path.relative_to(out_dir)),
                "train_samples": len(train_rows),
                "eval_samples": len(eval_rows),
                "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
                "eval_sha256": hashlib.sha256(eval_path.read_bytes()).hexdigest(),
            }
        )

    manifest_path = out_dir / "curriculum_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def _apply_merge(sequence: List[bytes], pair: Tuple[bytes, bytes]) -> List[bytes]:
    merged: List[bytes] = []
    i = 0
    while i < len(sequence):
        if i + 1 < len(sequence) and sequence[i] == pair[0] and sequence[i + 1] == pair[1]:
            merged.append(sequence[i] + sequence[i + 1])
            i += 2
        else:
            merged.append(sequence[i])
            i += 1
    return merged


def _is_bpe_whitespace_byte(value: int) -> bool:
    return chr(value).isspace()


def _is_bpe_word_byte(value: int) -> bool:
    return (
        48 <= value <= 57
        or 65 <= value <= 90
        or 97 <= value <= 122
        or value in (ord("_"), ord("-"), ord("/"))
        or value >= 0x80
    )


def _pretokenize_bytes_for_bpe(text: str) -> List[bytes]:
    raw = text.encode("utf-8")
    pieces: List[bytes] = []
    cursor = 0
    while cursor < len(raw):
        current = raw[cursor]
        if _is_bpe_whitespace_byte(current):
            cursor += 1
            while cursor < len(raw) and _is_bpe_whitespace_byte(raw[cursor]):
                cursor += 1
            continue

        word = _is_bpe_word_byte(current)
        end = cursor + 1
        if word:
            while end < len(raw) and _is_bpe_word_byte(raw[end]):
                end += 1
        pieces.append(raw[cursor:end])
        cursor = end
    return pieces


def learn_bpe_merges(texts: Sequence[str], target_vocab: int) -> List[Tuple[bytes, bytes]]:
    sequences: List[List[bytes]] = []
    for text in texts:
        for piece in _pretokenize_bytes_for_bpe(text):
            if len(piece) >= 2:
                sequences.append([bytes([value]) for value in piece])

    merges_needed = max(target_vocab - 256, 0)
    merges: List[Tuple[bytes, bytes]] = []
    for _rank in range(merges_needed):
        pair_counts: Counter = Counter()
        for sequence in sequences:
            for index in range(len(sequence) - 1):
                pair_counts[(sequence[index], sequence[index + 1])] += 1

        if not pair_counts:
            break
        pair, freq = pair_counts.most_common(1)[0]
        if freq < 2:
            break
        merges.append(pair)
        sequences = [_apply_merge(sequence, pair) for sequence in sequences]
    return merges


def write_ox3(path: Path, merges: Sequence[Tuple[bytes, bytes]]) -> None:
    with path.open("wb") as output:
        output.write(b"OX3\x00")
        output.write(struct.pack("<I", 1))
        for rank, (left, right) in enumerate(merges):
            output.write(struct.pack("<I", rank))
            output.write(struct.pack("<I", len(left)))
            output.write(left)
            output.write(struct.pack("<I", len(right)))
            output.write(right)


def _answer_is_natural_for_tokenizer(phase_name: str, row: Dict) -> bool:
    answer = row.get("answer", "").strip()
    if not answer:
        return False
    lowered = answer.lower()
    if any(marker in answer for marker in ["{", "}", "[", "]"]):
        return False
    if "," in answer and len(answer.split()) <= 4:
        return False
    if re.fullmatch(r"[0-9<>=+\-]+", answer):
        return False
    if lowered in {"status", "error", "warning", "agent", "cache", "edge", "train", "pt", "en"}:
        return False
    if phase_name in {"phase4_instructions", "phase6_memory"}:
        return True
    if len(answer) >= 5 and any(ch.isalpha() for ch in answer):
        return True
    return False


def _tokenizer_training_texts_for_row(phase_name: str, row: Dict) -> List[str]:
    if phase_name == "phase3_curated_text":
        text = row.get("text", "").strip()
        if not text:
            return []
        phase3_chunks = chunk_text(text, chunk_chars=640, overlap_chars=96)
        return phase3_chunks[: max(1, min(3, len(phase3_chunks)))]

    prompt = row.get("prompt", "").strip()
    answer = row.get("answer", "").strip()
    kind = row.get("kind", "")
    texts: List[str] = []

    if prompt:
        prompt_text = f"<|task:{kind}|>\nPrompt:\n{prompt[:220]}\nAnswer:\n"
        prompt_weight = 3 if phase_name in {"phase4_instructions", "phase6_memory"} else 2
        texts.extend([prompt_text] * prompt_weight)

    if answer and _answer_is_natural_for_tokenizer(phase_name, row):
        answer_weight = 2 if phase_name == "phase4_instructions" else 1
        texts.extend([answer[:140]] * answer_weight)

    return texts


def _downsample_texts(texts: List[str], max_items: int, max_chars: int) -> List[str]:
    if not texts:
        return []
    if len(texts) > max_items:
        stride = max(1, len(texts) // max_items)
        texts = texts[::stride][:max_items]
    trimmed: List[str] = []
    total_chars = 0
    for text in texts:
        if total_chars >= max_chars:
            break
        trimmed.append(text)
        total_chars += len(text)
    return trimmed


def build_tokenizer_bundle(curriculum_root: Path, target_vocab: int) -> Path:
    manifest = json.loads((curriculum_root / "curriculum_manifest.json").read_text(encoding="utf-8"))
    texts: List[str] = []
    phase_text_counts: Dict[str, int] = {}
    for phase in manifest["phases"]:
        phase_name = phase["name"]
        train_rows = read_jsonl(curriculum_root / phase["train_file"])
        eval_rows = read_jsonl(curriculum_root / phase["eval_file"])
        phase_texts: List[str] = []
        for row in train_rows:
            phase_texts.extend(_tokenizer_training_texts_for_row(phase_name, row))
        for row in eval_rows[: max(1, len(eval_rows) // 2)]:
            phase_texts.extend(_tokenizer_training_texts_for_row(phase_name, row))
        phase_texts = _downsample_texts(
            phase_texts,
            TOKENIZER_PHASE_TEXT_CAPS.get(phase_name, 512),
            TOKENIZER_PHASE_CHAR_BUDGETS.get(phase_name, 160_000),
        )
        texts.extend(phase_texts)
        phase_text_counts[phase_name] = len(phase_texts)

    merges = learn_bpe_merges(texts, target_vocab=target_vocab - len(SPECIAL_TOKENS))
    tokenizer_path = curriculum_root / f"tokenizer_{target_vocab}.ox3"
    write_ox3(tokenizer_path, merges)

    metadata = {
        "target_vocab": target_vocab,
        "base_vocab": 256,
        "special_tokens": SPECIAL_TOKENS,
        "learned_merges": len(merges),
        "tokenizer_training_texts": len(texts),
        "tokenizer_phase_weights": phase_text_counts,
        "sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
    }
    (curriculum_root / f"tokenizer_{target_vocab}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return tokenizer_path



def read_text_strict(path) -> str:
    """(auditoria #15) Leitura com contabilidade de perda: errors="ignore"
    descartava bytes em silencio.  Decodifica com errors="replace", conta os
    U+FFFD e ABORTA acima de 0.1% (corpus corrompido nao entra calado)."""
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8", errors="replace")
    bad = text.count(chr(0xFFFD))
    if bad and bad > max(1, len(text) // 1000):
        raise ValueError(f"{path}: {bad} bytes invalidos (> 0.1%) — corpus corrompido")
    return text


def curriculum_texts_for_phase(curriculum_root: Path, phase_name: str, split: str) -> List[Dict]:
    suffix = "train" if split == "train" else "eval"
    return read_jsonl(curriculum_root / "data" / f"{phase_name}.{suffix}.jsonl")
