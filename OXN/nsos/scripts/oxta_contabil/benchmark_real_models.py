"""Evaluate real pretrained checkpoints on a deterministic BR-TaxQA-R task.

The benchmark turns public Brazilian tax QA pairs into four-way multiple-choice
questions with hard, deterministic distractors.  It calls the local Ollama API
and records exact accuracy plus runtime telemetry.  This is an external baseline
for a future pretrained NSOS checkpoint; it must not be conflated with the
from-scratch bAbI architecture ablation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODELS = [
    "smollm2:135m-instruct-q4_K_M",
    "qwen2.5:0.5b-instruct-q4_K_M",
    "hf.co/tiiuae/Falcon-H1-0.5B-Instruct-GGUF:Q4_K_M",
]
LETTERS = "ABCD"
WORD_RE = re.compile(r"[a-z0-9]+", re.I)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root
        / "artifacts"
        / "oxta_contabil_amd"
        / "data"
        / "oxta_contabil"
        / "raw"
        / "br_taxqa"
        / "questions_QA_2024_v1.1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root
        / "artifacts"
        / "oxta_contabil_amd"
        / "benchmark"
        / "real_models_br_taxqa_mcq.json",
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--api", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "NSOS-benchmark/1"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "NSOS-benchmark/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize_text(value: str) -> str:
    return " ".join(str(value).split())


def answer_text(row: dict[str, Any]) -> str:
    value = row.get("answer_cleaned") or row.get("answer") or ""
    if isinstance(value, list):
        value = " ".join(str(item) for item in value if str(item).strip())
    return normalize_text(str(value))


def lexical_tokens(text: str) -> set[str]:
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return set(WORD_RE.findall(folded))


def stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def jaccard(lhs: set[str], rhs: set[str]) -> float:
    union = lhs | rhs
    return len(lhs & rhs) / len(union) if union else 0.0


def build_cases(rows: list[dict[str, Any]], samples: int) -> list[dict[str, Any]]:
    usable = [
        row
        for row in rows
        if normalize_text(row.get("question_text", ""))
        and len(answer_text(row)) >= 24
    ]
    if samples < 1 or samples > len(usable):
        raise ValueError(f"samples must be in 1..{len(usable)}")
    question_tokens = [lexical_tokens(row["question_text"]) for row in usable]
    order = sorted(
        range(len(usable)),
        key=lambda index: stable_hash(
            f"br-taxqa-eval-v1:{usable[index].get('question_number', index)}"
        ),
    )[:samples]
    cases: list[dict[str, Any]] = []
    for index in order:
        row = usable[index]
        summary = normalize_text(row.get("question_summary", "")).lower()
        ranked: list[tuple[float, int, int]] = []
        for other_index, other in enumerate(usable):
            if other_index == index:
                continue
            other_summary = normalize_text(other.get("question_summary", "")).lower()
            score = jaccard(question_tokens[index], question_tokens[other_index])
            if summary and summary == other_summary:
                score += 0.35
            tie = stable_hash(
                f"distractor:{row.get('question_number')}:{other.get('question_number')}"
            )
            ranked.append((score, tie, other_index))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        distractors = [item[2] for item in ranked[:3]]
        choices = [(index, answer_text(row))] + [
            (other_index, answer_text(usable[other_index]))
            for other_index in distractors
        ]
        rng = random.Random(stable_hash(f"choice-order:{row.get('question_number')}"))
        rng.shuffle(choices)
        correct_position = next(
            position for position, (source_index, _) in enumerate(choices)
            if source_index == index
        )
        excerpts = [text[:700].rstrip() for _, text in choices]
        prompt_lines = [
            "Você é um contador brasileiro. Escolha a alternativa que responde",
            "com maior precisão à pergunta tributária abaixo.",
            "",
            f"Pergunta: {normalize_text(row['question_text'])}",
            "",
        ]
        prompt_lines.extend(
            f"{letter}) {excerpt}"
            for letter, excerpt in zip(LETTERS, excerpts)
        )
        prompt_lines.extend(
            ["", "Responda SOMENTE uma letra: A, B, C ou D."]
        )
        cases.append(
            {
                "question_number": str(row.get("question_number", index)),
                "question_summary": normalize_text(row.get("question_summary", "")),
                "question": normalize_text(row["question_text"]),
                "choices": excerpts,
                "correct": LETTERS[correct_position],
                "prompt": "\n".join(prompt_lines),
            }
        )
    return cases


def parse_letter(response: str) -> str | None:
    text = response.strip().upper()
    leading = re.match(r"^\s*(?:RESPOSTA\s*[:\-]?\s*)?([ABCD])(?:\b|[\).:\-])", text)
    if leading:
        return leading.group(1)
    match = re.search(r"(?<![A-Z])([ABCD])(?![A-Z])", text)
    return match.group(1) if match else None


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def lexical_baseline(cases: list[dict[str, Any]]) -> dict[str, Any]:
    correct = 0
    predictions: list[str] = []
    for case in cases:
        question = lexical_tokens(case["question"])
        scores = [jaccard(question, lexical_tokens(choice)) for choice in case["choices"]]
        prediction = LETTERS[max(range(4), key=lambda index: (scores[index], -index))]
        predictions.append(prediction)
        correct += int(prediction == case["correct"])
    return {
        "name": "lexical_jaccard",
        "accuracy": correct / len(cases),
        "correct": correct,
        "total": len(cases),
        "wilson_95": wilson(correct, len(cases)),
        "predictions": predictions,
    }


def model_metadata(api: str, timeout: float) -> dict[str, dict[str, Any]]:
    tags = get_json(f"{api}/api/tags", timeout)
    return {row["name"]: row for row in tags.get("models", [])}


def loaded_model_metadata(api: str, model: str, timeout: float) -> dict[str, Any] | None:
    rows = get_json(f"{api}/api/ps", timeout).get("models", [])
    for row in rows:
        if row.get("name") == model or row.get("model") == model:
            return row
    return None


def evaluate_model(
    api: str,
    model: str,
    cases: list[dict[str, Any]],
    timeout: float,
) -> dict[str, Any]:
    options = {
        "temperature": 0.0,
        "seed": 20260728,
        "top_k": 1,
        "num_predict": 8,
        "num_ctx": 4096,
    }
    # Warm-up loads the checkpoint and compiles/initializes the runtime.
    post_json(
        f"{api}/api/generate",
        {
            "model": model,
            "prompt": cases[0]["prompt"],
            "stream": False,
            "keep_alive": "10m",
            "options": options,
        },
        timeout,
    )
    loaded = loaded_model_metadata(api, model, timeout)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for offset, case in enumerate(cases):
        response = post_json(
            f"{api}/api/generate",
            {
                "model": model,
                "prompt": case["prompt"],
                "stream": False,
                "keep_alive": "10m",
                "options": options,
            },
            timeout,
        )
        raw = str(response.get("response", ""))
        prediction = parse_letter(raw)
        record = {
            "question_number": case["question_number"],
            "correct": case["correct"],
            "predicted": prediction,
            "is_correct": prediction == case["correct"],
            "raw_response": raw,
            "total_duration_ns": int(response.get("total_duration", 0)),
            "load_duration_ns": int(response.get("load_duration", 0)),
            "prompt_eval_count": int(response.get("prompt_eval_count", 0)),
            "prompt_eval_duration_ns": int(response.get("prompt_eval_duration", 0)),
            "eval_count": int(response.get("eval_count", 0)),
            "eval_duration_ns": int(response.get("eval_duration", 0)),
        }
        records.append(record)
        if offset == 0 or (offset + 1) % 20 == 0:
            running = sum(int(item["is_correct"]) for item in records) / len(records)
            print(
                f"[{model}] {offset + 1}/{len(cases)} accuracy={running:.3f}",
                flush=True,
            )
    wall_seconds = time.perf_counter() - started
    correct = sum(int(record["is_correct"]) for record in records)
    invalid = sum(record["predicted"] is None for record in records)
    prompt_tokens = sum(record["prompt_eval_count"] for record in records)
    prompt_ns = sum(record["prompt_eval_duration_ns"] for record in records)
    generated_tokens = sum(record["eval_count"] for record in records)
    generation_ns = sum(record["eval_duration_ns"] for record in records)
    total_ns = sum(record["total_duration_ns"] for record in records)
    result = {
        "model": model,
        "accuracy": correct / len(records),
        "correct": correct,
        "total": len(records),
        "invalid_responses": invalid,
        "wilson_95": wilson(correct, len(records)),
        "wall_seconds": wall_seconds,
        "average_latency_ms": total_ns / len(records) / 1e6,
        "prompt_tokens_per_second": (
            prompt_tokens / (prompt_ns / 1e9) if prompt_ns > 0 else None
        ),
        "generation_tokens_per_second": (
            generated_tokens / (generation_ns / 1e9) if generation_ns > 0 else None
        ),
        "loaded_runtime": loaded,
        "records": records,
    }
    # Explicit unload keeps per-model VRAM measurements independent.
    post_json(
        f"{api}/api/generate",
        {"model": model, "prompt": "", "stream": False, "keep_alive": 0},
        timeout,
    )
    return result


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = build_cases(rows, args.samples)
    positions = {letter: sum(case["correct"] == letter for case in cases) for letter in LETTERS}
    print(f"cases={len(cases)} correct_position_counts={positions}", flush=True)
    metadata = model_metadata(args.api, args.timeout)
    missing = [model for model in args.models if model not in metadata]
    if missing:
        raise RuntimeError(f"models are not installed in Ollama: {missing}")

    results: list[dict[str, Any]] = []
    partial = args.report.resolve().with_suffix(".partial.json")
    partial.parent.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        result = evaluate_model(args.api, model, cases, args.timeout)
        result["installed_model"] = metadata[model]
        results.append(result)
        partial.write_text(
            json.dumps({"complete": False, "results": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"RESULT {model} accuracy={result['accuracy']:.3f} "
            f"latency_ms={result['average_latency_ms']:.1f}",
            flush=True,
        )

    report = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "name": "unicamp-dl/BR-TaxQA-R questions_QA_2024_v1.1",
            "path": str(dataset_path),
            "sha256": sha256(dataset_path),
            "source_rows": len(rows),
            "evaluated_rows": len(cases),
            "correct_position_counts": positions,
        },
        "protocol": {
            "task": "deterministic 4-way multiple choice with hard distractors",
            "language": "Portuguese (Brazil)",
            "temperature": 0.0,
            "seed": 20260728,
            "num_predict": 8,
            "num_ctx": 4096,
            "scoring": "exact parsed A/B/C/D; no LLM judge",
            "quantization": "Q4_K_M for all public checkpoints",
        },
        "baselines": {
            "random_accuracy": 0.25,
            "lexical": lexical_baseline(cases),
        },
        "results": results,
        "nsos_status": {
            "included_in_scoreboard": False,
            "reason": (
                "There is no pretrained Portuguese/accounting NSOS checkpoint. "
                "The earlier NSOS numbers came from task-specific training from random "
                "initialization and are not comparable to these pretrained checkpoints."
            ),
        },
        "cases": cases,
    }
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"REPORT {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
