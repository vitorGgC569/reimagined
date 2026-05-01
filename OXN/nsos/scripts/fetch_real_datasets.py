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
except Exception:  # pragma: no cover - optional runtime dependency
    duckdb = None

try:
    from huggingface_hub import HfApi
except Exception:  # pragma: no cover - optional runtime dependency
    HfApi = None


BASE_URL = "https://datasets-server.huggingface.co"

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
}


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split()).strip()


def fetch_json(path: str, params: Dict[str, object]) -> Dict:
    query = urllib.parse.urlencode(params)
    url = f"{BASE_URL}{path}?{query}"
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                return json.load(response)
        except Exception as exc:  # pragma: no cover - network instability
            last_error = exc
            if attempt == 4:
                break
            time.sleep(min(2 ** attempt, 8) + 0.25)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


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


def _select_entries(total_entries: int, pages: int) -> List[int]:
    if total_entries <= 1:
        return [0]
    max_index = max(total_entries - 1, 0)
    indices = set()
    for index in range(min(pages, total_entries)):
        ratio = index / max(min(pages, total_entries) - 1, 1)
        indices.add(min(int(round(ratio * max_index)), max_index))
    return sorted(indices)


def _download_parquet_entry(dataset: str, url: str, temp_dir: Path) -> Path:
    encoded_prefix = "/resolve/refs%2Fconvert%2Fparquet/"
    if encoded_prefix not in url:
        raise RuntimeError(f"Unexpected parquet URL format: {url}")
    relative_path = urllib.parse.unquote(url.split(encoded_prefix, 1)[1])
    local_path = temp_dir / relative_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, local_path)
        return local_path
    except Exception:
        command = [
            "hf",
            "download",
            dataset,
            relative_path.replace("\\", "/"),
            "--repo-type",
            "dataset",
            "--revision",
            "refs/convert/parquet",
            "--local-dir",
            str(temp_dir),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        if not local_path.exists():
            raise RuntimeError(f"hf download did not materialize expected parquet file: {local_path}")
        return local_path


def fetch_dataset_rows_via_parquet(name: str, spec: Dict[str, object]) -> List[Dict]:
    if HfApi is None or duckdb is None:
        raise RuntimeError("Parquet fallback unavailable because huggingface_hub or duckdb is missing.")

    api = HfApi()
    entries = [
        entry
        for entry in api.list_dataset_parquet_files(str(spec["dataset"]))
        if entry.config == str(spec["config"]) and entry.split == str(spec["split"])
    ]
    if not entries:
        raise RuntimeError(f"No parquet entries found for {spec['dataset']}:{spec['config']}:{spec['split']}")

    filter_fn = FILTERS[name]
    selected_entries = [entries[index] for index in _select_entries(len(entries), int(spec["pages"]))]
    kept: List[Dict] = []
    seen: set[str] = set()

    with tempfile.TemporaryDirectory() as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        for entry in selected_entries:
            parquet_path = _download_parquet_entry(str(spec["dataset"]), entry.url, temp_dir)
            conn = duckdb.connect()
            limit = max(int(spec["page_length"]) * 4, min(int(spec["target_rows"]) * 2, 4096))
            cursor = conn.execute("SELECT * FROM read_parquet(?) LIMIT ?", [str(parquet_path), limit])
            columns = [desc[0] for desc in cursor.description]
            for values in cursor.fetchall():
                row = dict(zip(columns, values))
                candidate = filter_fn(row)
                if candidate is None:
                    continue
                signature = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
                if signature in seen:
                    continue
                seen.add(signature)
                kept.append(candidate)
                if len(kept) >= int(spec["target_rows"]):
                    return kept
            time.sleep(0.1)
    return kept


def filter_wikitext(row: Dict) -> Dict | None:
    text = normalize_whitespace(row.get("text", ""))
    if len(text) < 240:
        return None
    if text.startswith("=") and text.endswith("="):
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
    if len(answer) > 48 or len(answer.split()) > 6:
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
    if len(answer) > 32 or len(answer.split()) > 4:
        return None
    return {
        "source": "ChilleD/SVAMP:default",
        "body": body,
        "question": question,
        "answer": answer,
        "equation": normalize_whitespace(row.get("Equation", "")),
        "type": normalize_whitespace(row.get("Type", "")),
    }


def _extract_final_math_answer(answer_text: str) -> str:
    clean = normalize_whitespace(answer_text)
    if not clean:
        return ""
    marker = "####"
    if marker in clean:
        tail = normalize_whitespace(clean.split(marker)[-1])
        if tail:
            return tail
    patterns = [
        r"(?i)(?:answer is|so,|thus,|therefore,|final answer:?)[^0-9-]*([-+]?\d+(?:\.\d+)?)",
        r"([-+]?\d+(?:\.\d+)?)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean)
        if match:
            return match.group(1)
    return ""


def filter_orca_math(row: Dict) -> Dict | None:
    question = normalize_whitespace(row.get("question", ""))
    answer = _extract_final_math_answer(str(row.get("answer", "")))
    if len(question) < 12 or not answer:
        return None
    if len(answer) > 24 or len(answer.split()) > 3:
        return None
    return {
        "source": "microsoft/orca-math-word-problems-200k:default",
        "question": question,
        "answer": answer,
    }


def filter_wikipedia(row: Dict, source_name: str) -> Dict | None:
    title = normalize_whitespace(row.get("title", ""))
    text = normalize_whitespace(row.get("text", ""))
    if len(text) < 360 or len(title) < 2:
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
    if len(text) < 320 or len(summary) < 24:
        return None
    return {
        "source": f"csebuetnlp/xlsum:{lang}",
        "lang": "pt" if lang == "portuguese" else "en",
        "title": title,
        "text": text,
        "summary": summary,
        "url": row.get("url", ""),
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


def filter_coqa(row: Dict) -> Dict | None:
    story = normalize_whitespace(row.get("story", ""))
    questions = row.get("questions", []) or []
    answers = (row.get("answers", {}) or {}).get("input_text", []) or []
    answer_starts = (row.get("answers", {}) or {}).get("answer_start", []) or []
    if len(story) < 180 or len(questions) < 3 or len(answers) < 3:
        return None
    pairs = []
    for question, answer, answer_start in zip(questions, answers, answer_starts):
        clean_question = normalize_whitespace(question)
        clean_answer = normalize_whitespace(answer)
        if len(clean_question) < 4 or not clean_answer:
            continue
        if clean_answer.lower() == "unknown" or clean_answer.lower() == "yes" or clean_answer.lower() == "no":
            continue
        if len(clean_answer) > 48 or len(clean_answer.split()) > 6:
            continue
        pairs.append(
            {
                "question": clean_question,
                "answer": clean_answer,
                "answer_start": int(answer_start) if isinstance(answer_start, int) else -1,
            }
        )
    if len(pairs) < 3:
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
    if len(english.split()) <= 2 or len(portuguese.split()) <= 2:
        return None
    return {
        "source": "Helsinki-NLP/opus_books:en-pt",
        "en": english,
        "pt": portuguese,
    }


FILTERS: Dict[str, Callable[[Dict], Dict | None]] = {
    "orca_math_word_problems": filter_orca_math,
    "deepmind_math_large": filter_deepmind_math,
    "svamp": filter_svamp,
    "triviaqa_rc_wikipedia": filter_triviaqa,
    "coqa": filter_coqa,
    "wikitext_en": filter_wikitext,
    "wikipedia_en": lambda row: filter_wikipedia(row, "wikimedia/wikipedia:20231101.en"),
    "wikipedia_pt": lambda row: filter_wikipedia(row, "wikimedia/wikipedia:20231101.pt"),
    "xlsum_en": lambda row: filter_xlsum(row, "english"),
    "xlsum_pt": lambda row: filter_xlsum(row, "portuguese"),
    "opus_books_en_pt": filter_opus,
}


def fetch_dataset_rows(name: str, spec: Dict[str, object]) -> List[Dict]:
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
    except Exception:
        return fetch_dataset_rows_via_parquet(name, spec)
    total_rows = int(initial.get("num_rows_total") or 0)
    offsets = choose_offsets(total_rows, page_length, int(spec["pages"]))
    filter_fn = FILTERS[name]
    kept: List[Dict] = []
    seen: set[str] = set()

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
            if len(kept) >= int(spec["target_rows"]):
                return kept
        time.sleep(0.1)
    return kept


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch small real datasets for NSOS curriculum phases.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to OXN/nsos/artifacts/real_datasets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.repo_root / "OXN" / "nsos" / "artifacts" / "real_datasets")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "base_url": BASE_URL,
        "datasets": {},
        "output_dir": str(out_dir),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    for name, spec in DATASET_SPECS.items():
        out_path = out_dir / f"{name}.jsonl"
        if out_path.exists() and not args.force:
            rows = sum(1 for _ in out_path.open("r", encoding="utf-8"))
            manifest["datasets"][name] = {
                "status": "kept_existing",
                "rows": rows,
                "dataset": spec["dataset"],
                "config": spec["config"],
                "split": spec["split"],
            }
            print(f"[skip] {name}: existing file with {rows} rows")
            continue

        try:
            rows = fetch_dataset_rows(name, spec)
            write_jsonl(out_path, rows)
            manifest["datasets"][name] = {
                "status": "downloaded",
                "rows": len(rows),
                "dataset": spec["dataset"],
                "config": spec["config"],
                "split": spec["split"],
            }
            print(f"[ok] {name}: {len(rows)} rows -> {out_path}")
        except Exception as exc:  # pragma: no cover - network instability
            manifest["datasets"][name] = {
                "status": "error",
                "error": str(exc),
                "dataset": spec["dataset"],
                "config": spec["config"],
                "split": spec["split"],
            }
            print(f"[error] {name}: {exc}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
