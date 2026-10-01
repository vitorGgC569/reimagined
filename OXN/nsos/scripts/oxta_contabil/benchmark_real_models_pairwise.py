"""Position-balanced BR-TaxQA-R benchmark for real pretrained checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

from benchmark_real_models import (
    DEFAULT_MODELS,
    answer_text,
    build_cases,
    get_json,
    jaccard,
    lexical_tokens,
    loaded_model_metadata,
    model_metadata,
    post_json,
    sha256,
    wilson,
)


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
        / "real_models_br_taxqa_pairwise.json",
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--api", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def parse_choice(response: str) -> str | None:
    text = response.strip()
    leading = re.match(
        r"^\s*(?:RESPOSTA|ALTERNATIVA|ANSWER)?\s*[:\-]?\s*([12])(?:\b|[\).:\-])",
        text,
        re.I,
    )
    if leading:
        return leading.group(1)
    match = re.search(r"(?<!\d)([12])(?!\d)", text)
    return match.group(1) if match else None


def comparison_prompt(question: str, first: str, second: str) -> str:
    return "\n".join(
        [
            f"Pergunta tributária: {question}",
            "",
            f"Resposta 1: {first}",
            "",
            f"Resposta 2: {second}",
            "",
            "Qual resposta é mais correta para a pergunta?",
            "Responda SOMENTE 1 ou 2.",
        ]
    )


def decisions_for_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for case in cases:
        correct_index = "ABCD".index(case["correct"])
        correct_answer = case["choices"][correct_index]
        distractors = [
            choice for index, choice in enumerate(case["choices"])
            if index != correct_index
        ]
        for distractor_index, distractor in enumerate(distractors):
            for correct_position in (0, 1):
                choices = (
                    [correct_answer, distractor]
                    if correct_position == 0
                    else [distractor, correct_answer]
                )
                decisions.append(
                    {
                        "question_number": case["question_number"],
                        "question": case["question"],
                        "distractor_index": distractor_index,
                        "correct_choice": str(correct_position + 1),
                        "prompt": comparison_prompt(
                            case["question"], choices[0], choices[1]
                        ),
                    }
                )
    return decisions


def lexical_baseline(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    correct = 0
    first_predictions = 0
    for decision in decisions:
        question, rest = decision["prompt"].split("\n\nResposta 1: ", 1)
        first, second = rest.split("\n\nResposta 2: ", 1)
        second = second.split("\n\nQual resposta", 1)[0]
        question_tokens = lexical_tokens(question)
        score_first = jaccard(question_tokens, lexical_tokens(first))
        score_second = jaccard(question_tokens, lexical_tokens(second))
        prediction = "1" if score_first >= score_second else "2"
        first_predictions += int(prediction == "1")
        correct += int(prediction == decision["correct_choice"])
    return {
        "accuracy": correct / len(decisions),
        "correct": correct,
        "total": len(decisions),
        "wilson_95": wilson(correct, len(decisions)),
        "first_position_rate": first_predictions / len(decisions),
    }


def evaluate_model(
    api: str,
    model: str,
    decisions: list[dict[str, Any]],
    timeout: float,
) -> dict[str, Any]:
    options = {
        "temperature": 0.0,
        "seed": 20260728,
        "top_k": 1,
        "num_predict": 8,
        "num_ctx": 4096,
    }

    def call(prompt: str) -> dict[str, Any]:
        return post_json(
            f"{api}/api/chat",
            {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Compare as duas respostas. Não continue o texto das "
                            "alternativas; devolva somente o número 1 ou 2."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "think": False,
                "keep_alive": "10m",
                "options": options,
            },
            timeout,
        )

    call(decisions[0]["prompt"])
    loaded = loaded_model_metadata(api, model, timeout)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, decision in enumerate(decisions):
        response = call(decision["prompt"])
        raw = str(response.get("message", {}).get("content", ""))
        prediction = parse_choice(raw)
        records.append(
            {
                "question_number": decision["question_number"],
                "distractor_index": decision["distractor_index"],
                "correct_choice": decision["correct_choice"],
                "predicted": prediction,
                "is_correct": prediction == decision["correct_choice"],
                "raw_response": raw,
                "total_duration_ns": int(response.get("total_duration", 0)),
                "prompt_eval_count": int(response.get("prompt_eval_count", 0)),
                "prompt_eval_duration_ns": int(
                    response.get("prompt_eval_duration", 0)
                ),
                "eval_count": int(response.get("eval_count", 0)),
                "eval_duration_ns": int(response.get("eval_duration", 0)),
            }
        )
        if index == 0 or (index + 1) % 60 == 0:
            running = sum(int(row["is_correct"]) for row in records) / len(records)
            print(
                f"[{model}] {index + 1}/{len(decisions)} accuracy={running:.3f}",
                flush=True,
            )
    wall_seconds = time.perf_counter() - started
    correct = sum(int(row["is_correct"]) for row in records)
    invalid = sum(row["predicted"] is None for row in records)
    first = sum(row["predicted"] == "1" for row in records)
    prompt_tokens = sum(row["prompt_eval_count"] for row in records)
    prompt_ns = sum(row["prompt_eval_duration_ns"] for row in records)
    output_tokens = sum(row["eval_count"] for row in records)
    output_ns = sum(row["eval_duration_ns"] for row in records)
    total_ns = sum(row["total_duration_ns"] for row in records)

    paired_wins = 0
    pairs = 0
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in records:
        key = (row["question_number"], row["distractor_index"])
        grouped.setdefault(key, []).append(row)
    for rows in grouped.values():
        if len(rows) == 2:
            pairs += 1
            paired_wins += int(all(row["is_correct"] for row in rows))

    result = {
        "model": model,
        "decision_accuracy": correct / len(records),
        "correct_decisions": correct,
        "total_decisions": len(records),
        "wilson_95": wilson(correct, len(records)),
        "invalid_responses": invalid,
        "first_position_rate": first / len(records),
        "paired_consistent_win_rate": paired_wins / pairs if pairs else 0.0,
        "paired_consistent_wins": paired_wins,
        "total_distractor_pairs": pairs,
        "wall_seconds": wall_seconds,
        "average_latency_ms": total_ns / len(records) / 1e6,
        "prompt_tokens_per_second": (
            prompt_tokens / (prompt_ns / 1e9) if prompt_ns else None
        ),
        "generation_tokens_per_second": (
            output_tokens / (output_ns / 1e9) if output_ns else None
        ),
        "loaded_runtime": loaded,
        "records": records,
    }
    post_json(
        f"{api}/api/generate",
        {"model": model, "prompt": "", "stream": False, "keep_alive": 0},
        timeout,
    )
    return result


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    rows = json.loads(dataset.read_text(encoding="utf-8"))
    cases = build_cases(rows, args.samples)
    decisions = decisions_for_cases(cases)
    if sum(row["correct_choice"] == "1" for row in decisions) * 2 != len(decisions):
        raise RuntimeError("pairwise decisions are not position-balanced")
    metadata = model_metadata(args.api, args.timeout)
    missing = [model for model in args.models if model not in metadata]
    if missing:
        raise RuntimeError(f"models are not installed in Ollama: {missing}")

    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    partial = report_path.with_suffix(".partial.json")
    results: list[dict[str, Any]] = []
    for model in args.models:
        result = evaluate_model(args.api, model, decisions, args.timeout)
        result["installed_model"] = metadata[model]
        results.append(result)
        partial.write_text(
            json.dumps({"complete": False, "results": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"RESULT {model} accuracy={result['decision_accuracy']:.3f} "
            f"position1={result['first_position_rate']:.3f}",
            flush=True,
        )

    report = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "name": "unicamp-dl/BR-TaxQA-R questions_QA_2024_v1.1",
            "path": str(dataset),
            "sha256": sha256(dataset),
            "source_rows": len(rows),
            "evaluated_questions": len(cases),
            "decisions_per_question": 6,
            "total_decisions_per_model": len(decisions),
        },
        "protocol": {
            "task": "pairwise correct-answer vs hard-distractor selection",
            "position_balancing": "every pair is evaluated in both orders",
            "language": "Portuguese (Brazil)",
            "temperature": 0.0,
            "seed": 20260728,
            "scoring": "exact 1/2; no LLM judge",
            "quantization": "Q4_K_M public checkpoints",
        },
        "baselines": {
            "random_decision_accuracy": 0.5,
            "lexical": lexical_baseline(decisions),
        },
        "results": results,
        "nsos_status": {
            "included_in_scoreboard": False,
            "reason": (
                "No pretrained Portuguese/accounting NSOS checkpoint exists yet. "
                "From-scratch task training is a different experiment."
            ),
        },
        "cases": cases,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"REPORT {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
