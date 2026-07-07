from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class TextPair:
    """One supervised prompt -> answer example."""

    prompt: str
    answer: str


class CharTokenizer:
    """Small deterministic char-level tokenizer for demos and probes.

    It is deliberately simple.  Production training should use the project's
    tokenizer pack, but this class is enough for synthetic tasks such as:
    "Quem é você?" -> "Oxta".
    """

    pad_token = "<PAD>"
    bos_token = "<BOS>"
    eos_token = "<EOS>"

    def __init__(self, tokens: Sequence[str]):
        unique = list(dict.fromkeys(tokens))
        required = [self.pad_token, self.bos_token, self.eos_token]
        merged = required + [t for t in unique if t not in required]
        self.tokens: List[str] = merged
        self.stoi = {tok: idx for idx, tok in enumerate(self.tokens)}
        self.itos = {idx: tok for tok, idx in self.stoi.items()}

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "CharTokenizer":
        chars = sorted({ch for text in texts for ch in text})
        return cls([cls.pad_token, cls.bos_token, cls.eos_token, *chars])

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.stoi[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.stoi[self.eos_token]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> List[int]:
        ids: List[int] = []
        if bos:
            ids.append(self.bos_id)
        for ch in text:
            if ch not in self.stoi:
                raise KeyError(f"character not in tokenizer vocabulary: {ch!r}")
            ids.append(self.stoi[ch])
        if eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], *, stop_at_eos: bool = True) -> str:
        out: List[str] = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id == self.eos_id and stop_at_eos:
                break
            if token_id in (self.pad_id, self.bos_id, self.eos_id):
                continue
            out.append(self.itos.get(token_id, ""))
        return "".join(out)


def load_text_pairs(pairs: Iterable[Tuple[str, str] | Mapping[str, str] | TextPair]) -> List[TextPair]:
    """Normalize in-memory examples into ``TextPair`` objects."""

    normalized: List[TextPair] = []
    for item in pairs:
        if isinstance(item, TextPair):
            normalized.append(item)
        elif isinstance(item, Mapping):
            normalized.append(TextPair(str(item["prompt"]), str(item["answer"])))
        else:
            prompt, answer = item
            normalized.append(TextPair(str(prompt), str(answer)))
    return normalized


def load_text_pairs_json(
    path: str | Path,
    *,
    prompt_key: str = "prompt",
    answer_key: str = "answer",
) -> List[TextPair]:
    """Load examples from a JSON file.

    Accepted shapes:

    - ``[{"prompt": "...", "answer": "..."}]``
    - ``{"data": [{"prompt": "...", "answer": "..."}]}``
    - ``{"examples": [{"prompt": "...", "answer": "..."}]}``
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, Mapping):
        if "data" in raw:
            raw = raw["data"]
        elif "examples" in raw:
            raw = raw["examples"]
    if not isinstance(raw, list):
        raise ValueError("JSON dataset must be a list or an object with data/examples")
    return [
        TextPair(prompt=str(row[prompt_key]), answer=str(row[answer_key]))
        for row in raw
    ]


def load_text_pairs_jsonl(
    path: str | Path,
    *,
    prompt_key: str = "prompt",
    answer_key: str = "answer",
) -> List[TextPair]:
    """Load one ``{"prompt": "...", "answer": "..."}`` object per line."""

    examples: List[TextPair] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        if not isinstance(row, Mapping):
            raise ValueError(f"JSONL line {line_no} is not an object")
        examples.append(TextPair(prompt=str(row[prompt_key]), answer=str(row[answer_key])))
    return examples


def pairs_to_token_ids(
    pairs: Sequence[TextPair],
    tokenizer: CharTokenizer,
    *,
    add_bos: bool = True,
    add_eos: bool = True,
) -> List[Tuple[List[int], List[int]]]:
    """Convert text pairs to NSOS supervised prompt/answer token lists."""

    return [
        (
            tokenizer.encode(pair.prompt, bos=add_bos, eos=False),
            tokenizer.encode(pair.answer, bos=False, eos=add_eos),
        )
        for pair in pairs
    ]
