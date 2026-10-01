from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Iterable, List

try:
    import duckdb
except Exception:
    duckdb = None

try:
    from huggingface_hub import HfApi
except Exception:
    HfApi = None

BASE_URL = "https://datasets-server.huggingface.co"

# ── REGEX SANITIZATION & BOILERPLATE REMOVAL (INDUSTRIAL QUALITY) ────

BOILERPLATE_PATTERNS = [
    r"(?i)\bleia\s+mais\b.*$",
    r"(?i)\bleia\s+també[mm]\b.*$",
    r"(?i)\bveja\s+també[mm]\b.*$",
    r"(?i)\bclique\s+aqui\b.*$",
    r"(?i)\binscreva-se\s+no\s+canal\b.*$",
    r"(?i)\bconfira\s+també[mm]\b.*$",
    r"(?i)\btodos\s+os\s+direitos\s+reservados\b.*$",
    r"(?i)\bpolítica\s+de\s+privacidade\b.*$",
    r"(?i)\btermo(?:s)?\s+de\s+uso\b.*$",
    r"(?i)\baceitar\s+all?\s+cookies?\b.*$",
    r"(?i)\bcompartilhe\s+(?:no|via)\s+(?:whatsapp|facebook|twitter|x|telegram)\b.*$",
    r"(?i)\bfoto:\s*reprodução.*$",
    r"(?i)\bimagem:\s*divulgação.*$",
    r"(?i)\bpublicado\s+em\s+\d{2}/\d{2}/\d{4}.*$",
    r"(?i)\bcrédito(?:s)?:\s*.*$",
    r"(?i)\bcarregando\s+comentários\b.*$",
    r"(?i)\bnavegação\s+de\s+artigo(?:s)?\b.*$",
    r"(?i)<[^>]+>",  # Residual HTML tags
    r"&nbsp;|&amp;|&quot;|&lt;|&gt;",  # HTML entities
]

COMPILED_BOILERPLATE = [re.compile(p) for p in BOILERPLATE_PATTERNS]


def sanitize_text_content(text: str) -> str:
    """Applies strict regex sanitization to remove web boilerplate,
    ads, navigation menus, and HTML residue.
    """
    if not text:
        return ""

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line_clean = line.strip()

        # Remove HTML residue
        line_clean = re.sub(r"<[^>]+>", "", line_clean)
        line_clean = re.sub(r"&nbsp;|&amp;|&quot;|&lt;|&gt;", " ", line_clean)

        # Check against boilerplate regex rules
        skip_line = False
        for pat in COMPILED_BOILERPLATE:
            if pat.search(line_clean):
                skip_line = True
                break

        if skip_line:
            continue

        if len(line_clean) > 0:
            cleaned_lines.append(line_clean)

    full_clean = " ".join(cleaned_lines)
    full_clean = re.sub(r"\s+", " ", full_clean).strip()
    return full_clean


def is_high_quality_text(text: str) -> bool:
    """Heuristic quality gate: filters out low-quality web dumps,
    uppercase-heavy text, navigation menus, and code residue.
    """
    if len(text) < 180 or len(text) > 8000:
        return False

    # Check uppercase ratio (menu / header dumps)
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    if uppercase_ratio > 0.22:
        return False

    # Check punctuation density
    words = text.split()
    if len(words) < 25:
        return False

    # Check unique word ratio (repetitive artifacts)
    unique_ratio = len(set(w.lower() for w in words)) / len(words)
    if unique_ratio < 0.35:
        return False

    return True


DATASET_SPECS = {
    "orca_math_word_problems": {
        "dataset": "microsoft/orca-math-word-problems-200k",
        "config": "default",
        "split": "train",
        "target_rows": 1600,
        "page_length": 100,
        "pages": 12,
    },
    "deepmind_math_large": {
        "dataset": "davidheineman/deepmind-math-large",
        "config": "default",
        "split": "train",
        "target_rows": 1600,
        "page_length": 100,
        "pages": 24,
    },
    "svamp": {
        "dataset": "ChilleD/SVAMP",
        "config": "default",
        "split": "train",
        "target_rows": 600,
        "page_length": 100,
        "pages": 8,
    },
    "triviaqa_rc_wikipedia": {
        "dataset": "mandarjoshi/trivia_qa",
        "config": "rc.wikipedia",
        "split": "train",
        "target_rows": 1400,
        "page_length": 100,
        "pages": 20,
    },
    "coqa": {
        "dataset": "stanfordnlp/coqa",
        "config": "default",
        "split": "train",
        "target_rows": 900,
        "page_length": 100,
        "pages": 12,
    },
    "wikitext_en": {
        "dataset": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "target_rows": 260,
        "page_length": 100,
        "pages": 10,
    },
    "wikipedia_en": {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.en",
        "split": "train",
        "target_rows": 220,
        "page_length": 100,
        "pages": 8,
    },
    "wikipedia_pt": {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.pt",
        "split": "train",
        "target_rows": 220,
        "page_length": 100,
        "pages": 8,
    },
    "xlsum_en": {
        "dataset": "csebuetnlp/xlsum",
        "config": "english",
        "split": "train",
        "target_rows": 180,
        "page_length": 100,
        "pages": 6,
    },
    "xlsum_pt": {
        "dataset": "csebuetnlp/xlsum",
        "config": "portuguese",
        "split": "train",
        "target_rows": 180,
        "page_length": 100,
        "pages": 6,
    },
    "opus_books_en_pt": {
        "dataset": "Helsinki-NLP/opus_books",
        "config": "en-pt",
        "split": "train",
        "target_rows": 220,
        "page_length": 100,
        "pages": 6,
    },
    "cosmopedia_v2": {
        "dataset": "HuggingFaceTB/cosmopedia",
        "config": "auto_math_text",
        "split": "train",
        "target_rows": 4000,
        "page_length": 100,
        "pages": 40,
    },
    "tinystories": {
        "dataset": "roneneldan/TinyStories",
        "config": "default",
        "split": "train",
        "target_rows": 5000,
        "page_length": 100,
        "pages": 50,
    },
    "smoltalk": {
        "dataset": "HuggingFaceTB/smoltalk",
        "config": "all",
        "split": "train",
        "target_rows": 4000,
        "page_length": 100,
        "pages": 40,
    },
    "the_stack_smol": {
        "dataset": "bigcode/the-stack-smol",
        "config": "data/python",
        "split": "train",
        "target_rows": 3000,
        "page_length": 100,
        "pages": 30,
    },
    "c4_sample": {
        "dataset": "allenai/c4",
        "config": "en",
        "split": "train",
        "target_rows": 2000,
        "page_length": 100,
        "pages": 20,
    },
    "squad_v2": {
        "dataset": "rajpurkar/squad_v2",
        "config": "squad_v2",
        "split": "train",
        "target_rows": 2000,
        "page_length": 100,
        "pages": 20,
    },
}


def normalize_whitespace(text: str) -> str:
    return sanitize_text_content(text)


def filter_wikitext(row: Dict) -> Dict | None:
    text = normalize_whitespace(row.get("text", ""))
    if not is_high_quality_text(text):
        return None
    return {
        "source": "Salesforce/wikitext:wikitext-103-raw-v1",
        "title": row.get("title", "WikiText passage"),
        "text": text,
    }


def filter_deepmind_math(row: Dict) -> Dict | None:
    question = normalize_whitespace(row.get("question", ""))
    answer = normalize_whitespace(str(row.get("answer", "")))
    if len(question) < 12 or not answer:
        return None
    return {
        "source": "davidheineman/deepmind-math-large:default",
        "question_type": row.get("question-type", ""),
        "subcategory": row.get("subcategory", ""),
        "question": question,
        "answer": answer,
    }


def filter_svamp(row: Dict) -> Dict | None:
    body = normalize_whitespace(row.get("Body", ""))
    question = normalize_whitespace(row.get("Question", ""))
    answer = normalize_whitespace(str(row.get("Answer", "")))
    if len(body) < 12 or len(question) < 8 or not answer:
        return None
    return {
        "source": "ChilleD/SVAMP:default",
        "body": body,
        "question": question,
        "answer": answer,
        "equation": normalize_whitespace(row.get("Equation", "")),
        "type": normalize_whitespace(row.get("Type", "")),
    }


def filter_orca_math(row: Dict) -> Dict | None:
    question = normalize_whitespace(row.get("question", ""))
    answer = normalize_whitespace(str(row.get("answer", "")))
    if len(question) < 12 or not answer:
        return None
    return {
        "source": "microsoft/orca-math-word-problems-200k:default",
        "question": question,
        "answer": answer,
    }


def filter_wikipedia(row: Dict, source_name: str) -> Dict | None:
    title = normalize_whitespace(row.get("title", ""))
    text = normalize_whitespace(row.get("text", ""))
    if not is_high_quality_text(text):
        return None
    return {
        "source": source_name,
        "title": title,
        "text": text,
        "url": row.get("url", ""),
    }


def filter_xlsum(row: Dict, lang: str) -> Dict | None:
    title = normalize_whitespace(row.get("title", ""))
    text = normalize_whitespace(row.get("text", ""))
    summary = normalize_whitespace(row.get("summary", ""))
    if not is_high_quality_text(text) or len(summary) < 20:
        return None
    return {
        "source": f"csebuetnlp/xlsum:{lang}",
        "lang": "pt" if lang == "portuguese" else "en",
        "title": title,
        "text": text,
        "summary": summary,
        "url": row.get("url", ""),
    }


def filter_coqa(row: Dict) -> Dict | None:
    story = normalize_whitespace(row.get("story", ""))
    questions = row.get("questions", []) or []
    answers = (row.get("answers", {}) or {}).get("input_text", []) or []
    if len(story) < 180 or len(questions) < 2:
        return None
    pairs = []
    for q, a in zip(questions, answers):
        cq = normalize_whitespace(q)
        ca = normalize_whitespace(a)
        if len(cq) >= 4 and len(ca) >= 1:
            pairs.append({"question": cq, "answer": ca})
    if not pairs:
        return None
    return {
        "source": "stanfordnlp/coqa:default",
        "story": story,
        "questions": pairs,
    }


def filter_opus(row: Dict) -> Dict | None:
    translation = row.get("translation", {})
    english = normalize_whitespace(translation.get("en", ""))
    portuguese = normalize_whitespace(translation.get("pt", ""))
    if len(english) < 16 or len(portuguese) < 16:
        return None
    return {
        "source": "Helsinki-NLP/opus_books:en-pt",
        "en": english,
        "pt": portuguese,
    }


def filter_cosmopedia(row: Dict) -> Dict | None:
    text = normalize_whitespace(row.get("text", ""))
    if not is_high_quality_text(text):
        return None
    title = normalize_whitespace(row.get("prompt", "")) or "Cosmopedia passage"
    return {
        "source": "HuggingFaceTB/cosmopedia:auto_math_text",
        "title": title,
        "text": text,
    }


def filter_tinystories(row: Dict) -> Dict | None:
    text = normalize_whitespace(row.get("text", ""))
    if not is_high_quality_text(text):
        return None
    return {
        "source": "roneneldan/TinyStories:default",
        "title": "TinyStory",
        "text": text,
    }


def filter_smoltalk(row: Dict) -> Dict | None:
    messages = row.get("messages", []) or []
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    user_msg = next((m for m in messages if m.get("role") == "user"), None)
    asst_msg = next((m for m in messages if m.get("role") == "assistant"), None)
    if not user_msg or not asst_msg:
        return None
    prompt = normalize_whitespace(user_msg.get("content", ""))
    answer = normalize_whitespace(asst_msg.get("content", ""))
    if len(prompt) < 6 or len(answer) < 6:
        return None
    return {
        "source": "HuggingFaceTB/smoltalk:all",
        "prompt": prompt,
        "answer": answer,
    }


def filter_the_stack_smol(row: Dict) -> Dict | None:
    content = row.get("content", "") or ""
    if len(content) < 80 or len(content) > 4000:
        return None
    return {
        "source": "bigcode/the-stack-smol:python",
        "path": row.get("path", ""),
        "content": content,
    }


def filter_c4(row: Dict) -> Dict | None:
    text = normalize_whitespace(row.get("text", ""))
    if not is_high_quality_text(text):
        return None
    return {
        "source": "allenai/c4:en",
        "url": row.get("url", ""),
        "text": text,
    }


def filter_squad_v2(row: Dict) -> Dict | None:
    question = normalize_whitespace(row.get("question", ""))
    context = normalize_whitespace(row.get("context", ""))
    answers = (row.get("answers", {}) or {}).get("text", []) or []
    if not answers:
        return None
    answer = normalize_whitespace(answers[0])
    if len(question) < 8 or len(context) < 40 or not answer:
        return None
    return {
        "source": "rajpurkar/squad_v2:squad_v2",
        "question": question,
        "context": context,
        "answer": answer,
    }


def _trim_context_window(text: str, center: int, radius: int = 240) -> str:
    if not text:
        return ""
    start = max(center - radius, 0)
    end = min(center + radius, len(text))
    if start > 0:
        split = text.rfind(" ", 0, start)
        if split > 0:
            start = split + 1
    if end < len(text):
        split = text.find(" ", end)
        if split > 0:
            end = split
    return normalize_whitespace(text[start:end])


def filter_triviaqa(row: Dict) -> Dict | None:
    question = normalize_whitespace(row.get("question", ""))
    answer = row.get("answer", {}) or {}
    answer_value = normalize_whitespace(answer.get("value", ""))
    if len(question) < 12 or not answer_value:
        return None
    if len(answer_value) > 40 or len(answer_value.split()) > 5:
        return None

    aliases = [answer_value]
    aliases.extend(answer.get("aliases", []) or [])
    aliases = [normalize_whitespace(alias) for alias in aliases if normalize_whitespace(alias)]

    entity_pages = row.get("entity_pages", {}) or {}
    titles = entity_pages.get("title", []) or []
    contexts = entity_pages.get("wiki_context", []) or []
    if not contexts:
        return None

    chosen_title = ""
    chosen_context = ""
    for index, context in enumerate(contexts):
        lowered = context.lower()
        match_index = -1
        for alias in aliases:
            match_index = lowered.find(alias.lower())
            if match_index >= 0:
                chosen_title = titles[index] if index < len(titles) else ""
                chosen_context = _trim_context_window(context, match_index)
                break
        if chosen_context:
            break
    if not chosen_context:
        return None

    return {
        "source": "mandarjoshi/trivia_qa:rc.wikipedia",
        "title": normalize_whitespace(chosen_title),
        "question": question,
        "answer": answer_value,
        "context": chosen_context,
    }


FILTERS = {
    "orca_math_word_problems": filter_orca_math,
    "deepmind_math_large": filter_deepmind_math,
    "svamp": filter_svamp,
    "triviaqa_rc_wikipedia": filter_triviaqa,
    "coqa": filter_coqa,
    "wikitext_en": filter_wikitext,
    "wikipedia_en": lambda r: filter_wikipedia(r, "wikimedia/wikipedia:20231101.en"),
    "wikipedia_pt": lambda r: filter_wikipedia(r, "wikimedia/wikipedia:20231101.pt"),
    "xlsum_en": lambda r: filter_xlsum(r, "english"),
    "xlsum_pt": lambda r: filter_xlsum(r, "portuguese"),
    "opus_books_en_pt": filter_opus,
    "cosmopedia_v2": filter_cosmopedia,
    "tinystories": filter_tinystories,
    "smoltalk": filter_smoltalk,
    "the_stack_smol": filter_the_stack_smol,
    "c4_sample": filter_c4,
    "squad_v2": filter_squad_v2,
}


def choose_offsets(total_rows: int, page_length: int, pages: int) -> List[int]:
    if total_rows <= page_length or pages <= 1:
        return [0]
    max_offset = max(total_rows - page_length, 0)
    offsets = set()
    for index in range(pages):
        ratio = index / max(pages - 1, 1)
        raw = int(round(ratio * max_offset))
        aligned = min((raw // page_length) * page_length, max_offset)
        offsets.add(max(0, aligned))
    return sorted(offsets)


def fetch_json(path: str, params: Dict[str, object]) -> Dict:
    query = urllib.parse.urlencode(params)
    url = f"{BASE_URL}{path}?{query}"
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                return json.load(response)
        except Exception as exc:
            last_error = exc
            if attempt == 4:
                break
            time.sleep(min(2 ** attempt, 8) + 0.25)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


import datasets


def fetch_dataset_rows(name: str, spec: Dict[str, object]) -> List[Dict]:
    filter_fn = FILTERS[name]
    target_rows = int(spec["target_rows"])
    kept: List[Dict] = []
    seen: set[str] = set()

    # Try streaming direct from HuggingFace Hub (bypasses /rows API rate limits)
    try:
        print(f"[fetch] Baixando via HuggingFace Streaming: {spec['dataset']} ({spec['config']})...")
        ds = datasets.load_dataset(
            str(spec["dataset"]),
            str(spec["config"]),
            split=str(spec["split"]),
            streaming=True,
        )
        for row in ds:
            candidate = filter_fn(row)
            if candidate is None:
                continue
            signature = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
            if signature in seen:
                continue
            seen.add(signature)
            kept.append(candidate)
            if len(kept) >= target_rows:
                print(f"[fetch] Sucesso! {len(kept)} amostras filtradas para {name}.")
                return kept
    except Exception as exc:
        print(f"[warn] Streaming fallback via HF API para {name}: {exc}")

    # Fallback to HTTP rows API
    page_length = int(spec["page_length"])
    try:
        initial = fetch_json(
            "/rows",
            {
                "dataset": spec["dataset"],
                "config": spec["config"],
                "split": spec["split"],
                "offset": 0,
                "length": page_length,
            },
        )
        total_rows = int(initial.get("num_rows_total") or 0)
        offsets = choose_offsets(total_rows, page_length, int(spec["pages"]))
        for offset in offsets:
            payload = initial if offset == 0 else fetch_json(
                "/rows",
                {
                    "dataset": spec["dataset"],
                    "config": spec["config"],
                    "split": spec["split"],
                    "offset": offset,
                    "length": page_length,
                },
            )
            for wrapped in payload.get("rows", []):
                candidate = filter_fn(wrapped.get("row", {}))
                if candidate is None:
                    continue
                signature = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
                if signature in seen:
                    continue
                seen.add(signature)
                kept.append(candidate)
                if len(kept) >= target_rows:
                    return kept
            time.sleep(0.1)
    except Exception as exc:
        print(f"[warn] HTTP API fallback para {name}: {exc}")

    return kept


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and SANITIZE clean datasets for NSOS.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.repo_root / "OXN" / "nsos" / "artifacts" / "real_datasets_clean")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[clean_dataset] Dataset sanitization script ready. Target dir: {out_dir}")


if __name__ == "__main__":
    main()

