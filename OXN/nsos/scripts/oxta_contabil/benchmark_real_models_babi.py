"""Evaluate real Ollama checkpoints on the exact NSOS bAbI QA1 holdout.

This harness is intentionally external to the NSOS runtime. It uses only the
Ollama HTTP API to treat public checkpoints as black boxes; no public-model
framework or weight-loading dependency is linked into the authorial C++/HIP
product.

The shared comparison contract is the immutable bAbI test file, all 1,000
holdout examples, deterministic decoding, a one-token answer target and exact
normalized scoring without an LLM judge. Training histories remain materially
different and are recorded as such: NSOS is trained end to end from random
initialization, while the public models are pretrained Q4_K_M checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import unicodedata
from pathlib import Path
from typing import Any

from benchmark_product_architecture import (
    BABI_REVISION,
    atomic_write_json,
    git_provenance,
    load_babi,
)
from benchmark_real_models import (
    DEFAULT_MODELS,
    get_json,
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
        "--data-dir",
        type=Path,
        default=root / "artifacts" / "oxta_contabil_amd" / "data",
    )
    parser.add_argument(
        "--nsos-report",
        type=Path,
        default=(
            root
            / "artifacts"
            / "oxta_contabil_amd"
            / "benchmark"
            / "rx7600_deterministic_architecture1500_3seed_2026-07-29.json"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            root
            / "artifacts"
            / "oxta_contabil_amd"
            / "benchmark"
            / "real_models_babi_same_holdout_2026-07-29.json"
        ),
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="0 evaluates the complete 1,000-example holdout.",
    )
    parser.add_argument(
        "--determinism-samples",
        type=int,
        default=100,
        help="Evenly spaced cases repeated after the full pass.",
    )
    parser.add_argument("--api", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_token(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value.strip().lower())
    return "".join(
        character for character in folded
        if not unicodedata.combining(character)
    )


def parse_answer(response: str, output_vocabulary: set[str]) -> str | None:
    try:
        payload = json.loads(response)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(
        payload.get("answer"), str
    ):
        return None
    token = normalize_token(payload["answer"])
    return token if token in output_vocabulary else None


def response_schema(output_vocabulary: set[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "enum": sorted(output_vocabulary),
            }
        },
        "required": ["answer"],
        "additionalProperties": False,
    }


def format_prompt(example: dict[str, Any]) -> str:
    facts = "\n".join(
        f"{index + 1}. {str(sentence).strip()}"
        for index, sentence in enumerate(example["context"])
    )
    return "\n".join(
        [
            "Read the facts and answer the question using exactly one "
            "lowercase English word.",
            "Do not explain the answer.",
            "Return only a JSON object with an answer field.",
            "",
            "Facts:",
            facts,
            "",
            f"Question: {str(example['question']).strip()}",
            "Answer:",
        ]
    )


def evenly_spaced_indices(total: int, requested: int) -> list[int]:
    if total < 1:
        raise ValueError("cannot sample an empty collection")
    if requested < 1 or requested >= total:
        return list(range(total))
    return [index * total // requested for index in range(requested)]


def build_cases(dataset: dict[str, Any], samples: int) -> list[dict[str, Any]]:
    examples = dataset["test_examples"]
    indices = evenly_spaced_indices(len(examples), samples)
    cases = []
    for test_index in indices:
        example = examples[test_index]
        answer = normalize_token(str(example["answer"]))
        if answer not in dataset["stoi"]:
            raise RuntimeError(
                f"test answer is outside train vocabulary at {test_index}: "
                f"{answer!r}"
            )
        prompt = format_prompt(example)
        case = {
            "test_index": test_index,
            "context": [str(value) for value in example["context"]],
            "question": str(example["question"]),
            "expected": answer,
            "prompt": prompt,
        }
        case["sha256"] = canonical_sha256(case)
        cases.append(case)
    return cases


def generation_options() -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "seed": 20260729,
        "top_k": 1,
        "num_predict": 16,
        "num_ctx": 1024,
    }


def generate_case(
    api: str,
    model: str,
    case: dict[str, Any],
    output_vocabulary: set[str],
    timeout: float,
) -> dict[str, Any]:
    response = post_json(
        f"{api}/api/generate",
        {
            "model": model,
            "prompt": case["prompt"],
            "format": response_schema(output_vocabulary),
            "stream": False,
            "keep_alive": "10m",
            "options": generation_options(),
        },
        timeout,
    )
    raw = str(response.get("response", ""))
    predicted = parse_answer(raw, output_vocabulary)
    return {
        "test_index": case["test_index"],
        "case_sha256": case["sha256"],
        "expected": case["expected"],
        "predicted": predicted,
        "is_correct": predicted == case["expected"],
        "raw_response": raw,
        "raw_response_sha256": hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest(),
        "total_duration_ns": int(response.get("total_duration", 0)),
        "load_duration_ns": int(response.get("load_duration", 0)),
        "prompt_eval_count": int(response.get("prompt_eval_count", 0)),
        "prompt_eval_duration_ns": int(
            response.get("prompt_eval_duration", 0)
        ),
        "eval_count": int(response.get("eval_count", 0)),
        "eval_duration_ns": int(response.get("eval_duration", 0)),
    }


def aggregate_records(
    model: str,
    records: list[dict[str, Any]],
    wall_seconds: float,
    loaded_runtime: dict[str, Any] | None,
) -> dict[str, Any]:
    total = len(records)
    correct = sum(int(record["is_correct"]) for record in records)
    invalid = sum(record["predicted"] is None for record in records)
    prompt_tokens = sum(record["prompt_eval_count"] for record in records)
    prompt_ns = sum(
        record["prompt_eval_duration_ns"] for record in records
    )
    generated_tokens = sum(record["eval_count"] for record in records)
    generation_ns = sum(record["eval_duration_ns"] for record in records)
    total_ns = sum(record["total_duration_ns"] for record in records)
    return {
        "model": model,
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "invalid_responses": invalid,
        "wilson_95": wilson(correct, total),
        "wall_seconds": wall_seconds,
        "examples_per_second": total / max(wall_seconds, 1e-12),
        "average_latency_ms": total_ns / total / 1e6,
        "prompt_tokens_per_second": (
            prompt_tokens / (prompt_ns / 1e9) if prompt_ns > 0 else None
        ),
        "generation_tokens_per_second": (
            generated_tokens / (generation_ns / 1e9)
            if generation_ns > 0
            else None
        ),
        "prediction_sha256": canonical_sha256(
            [
                {
                    "test_index": record["test_index"],
                    "predicted": record["predicted"],
                    "is_correct": record["is_correct"],
                }
                for record in records
            ]
        ),
        "loaded_runtime": loaded_runtime,
        "records": records,
    }


def unload_model(api: str, model: str, timeout: float) -> None:
    post_json(
        f"{api}/api/generate",
        {
            "model": model,
            "prompt": "",
            "stream": False,
            "keep_alive": 0,
        },
        timeout,
    )


def replay_sequence(
    api: str,
    model: str,
    cases: list[dict[str, Any]],
    output_vocabulary: set[str],
    timeout: float,
    repetition: int,
) -> list[dict[str, Any]]:
    unload_model(api, model, timeout)
    post_json(
        f"{api}/api/generate",
        {
            "model": model,
            "prompt": cases[0]["prompt"],
            "format": response_schema(output_vocabulary),
            "stream": False,
            "keep_alive": "10m",
            "options": generation_options(),
        },
        timeout,
    )
    records = []
    try:
        for offset, case in enumerate(cases):
            records.append(
                generate_case(
                    api, model, case, output_vocabulary, timeout
                )
            )
            if offset == 0 or (offset + 1) % 50 == 0:
                print(
                    f"[{model} replay={repetition}] "
                    f"{offset + 1}/{len(cases)}",
                    flush=True,
                )
        return records
    finally:
        unload_model(api, model, timeout)


def evaluate_model(
    api: str,
    model: str,
    cases: list[dict[str, Any]],
    output_vocabulary: set[str],
    timeout: float,
    determinism_samples: int,
    partial_path: Path,
    completed_results: list[dict[str, Any]],
) -> dict[str, Any]:
    options = generation_options()
    unload_model(api, model, timeout)
    post_json(
        f"{api}/api/generate",
        {
            "model": model,
            "prompt": cases[0]["prompt"],
            "format": response_schema(output_vocabulary),
            "stream": False,
            "keep_alive": "10m",
            "options": options,
        },
        timeout,
    )
    loaded = loaded_model_metadata(api, model, timeout)
    records = []
    started = time.perf_counter()
    try:
        for offset, case in enumerate(cases):
            records.append(
                generate_case(
                    api, model, case, output_vocabulary, timeout
                )
            )
            if offset == 0 or (offset + 1) % 50 == 0:
                running = (
                    sum(int(item["is_correct"]) for item in records)
                    / len(records)
                )
                print(
                    f"[{model}] {offset + 1}/{len(cases)} "
                    f"accuracy={running:.3%}",
                    flush=True,
                )
                atomic_write_json(
                    partial_path,
                    {
                        "complete": False,
                        "active_model": model,
                        "processed": offset + 1,
                        "active_records": records,
                        "completed_results": completed_results,
                    },
                )
        wall_seconds = time.perf_counter() - started
        result = aggregate_records(model, records, wall_seconds, loaded)

    finally:
        unload_model(api, model, timeout)

    repeat_indices = evenly_spaced_indices(
        len(cases), min(determinism_samples, len(cases))
    )
    replay_cases = [cases[index] for index in repeat_indices]
    replay_runs = [
        replay_sequence(
            api,
            model,
            replay_cases,
            output_vocabulary,
            timeout,
            repetition,
        )
        for repetition in (1, 2)
    ]
    prediction_exact = all(
        first["predicted"] == second["predicted"]
        for first, second in zip(*replay_runs)
    )
    raw_exact = all(
        first["raw_response"] == second["raw_response"]
        for first, second in zip(*replay_runs)
    )
    result["determinism_replay"] = {
        "samples": len(replay_cases),
        "indices": repeat_indices,
        "fresh_model_runs": 2,
        "prediction_exact": prediction_exact,
        "raw_response_exact": raw_exact,
        "prediction_sha256": [
            canonical_sha256(
                [record["predicted"] for record in replay]
            )
            for replay in replay_runs
        ],
        "raw_response_sha256": [
            canonical_sha256(
                [record["raw_response"] for record in replay]
            )
            for replay in replay_runs
        ],
    }
    if not prediction_exact:
        atomic_write_json(
            partial_path,
            {
                "complete": False,
                "failure": "deterministic prediction replay",
                "active_model": model,
                "result": result,
                "replay_runs": replay_runs,
                "completed_results": completed_results,
            },
        )
        raise RuntimeError(
            f"deterministic prediction replay failed for {model}"
        )
    return result


def load_nsos_results(
    path: Path,
    dataset: dict[str, Any],
    full_holdout: bool,
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    protocol = report.get("protocol", {})
    checks = {
        "complete": report.get("complete") is True,
        "dataset_revision": protocol.get("revision") == BABI_REVISION,
        "dataset_hashes": protocol.get("hashes") == dataset["hashes"],
        "test_examples": int(protocol.get("test_examples", -1))
        == len(dataset["test"]),
        "external_uses_full_holdout": full_holdout,
    }
    if full_holdout and not all(checks.values()):
        raise RuntimeError(
            f"NSOS/public-model protocol mismatch: {checks}"
        )
    rows = []
    for result in report.get("results", []):
        rows.append(
            {
                "model": (
                    f"nsos/{result['arm']}/seed-{int(result['seed'])}"
                ),
                "arm": result["arm"],
                "seed": int(result["seed"]),
                "accuracy": float(result["accuracy"]),
                "correct": round(
                    float(result["accuracy"]) * len(dataset["test"])
                ),
                "total": len(dataset["test"]),
                "adaptation_scope": "end_to_end_from_random_initialization",
                "trainable_parameters": int(
                    result["trainable_parameters"]
                ),
                "values_sha256": result["final_parameter_manifest"][
                    "values_sha256"
                ],
            }
        )
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "protocol_checks": checks,
        "results": rows,
    }


def main() -> int:
    args = parse_args()
    if (
        args.samples < 0
        or args.determinism_samples < 1
        or args.timeout <= 0
        or not args.models
        or len(set(args.models)) != len(args.models)
    ):
        raise ValueError("invalid or duplicated benchmark dimensions")

    repository_root = Path(__file__).resolve().parents[4]
    provenance = git_provenance(
        repository_root, args.require_clean_git
    )
    dataset = load_babi(args.data_dir)
    cases = build_cases(dataset, args.samples)
    full_holdout = len(cases) == len(dataset["test"])
    output_vocabulary = {
        normalize_token(str(token)) for token in dataset["vocab"]
    }
    metadata = model_metadata(args.api, args.timeout)
    missing = [model for model in args.models if model not in metadata]
    if missing:
        raise RuntimeError(
            f"models are not installed in Ollama: {missing}"
        )

    report_path = args.report.resolve()
    partial_path = report_path.with_suffix(".partial.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for model in args.models:
        result = evaluate_model(
            args.api,
            model,
            cases,
            output_vocabulary,
            args.timeout,
            args.determinism_samples,
            partial_path,
            results,
        )
        result["installed_model"] = metadata[model]
        results.append(result)
        atomic_write_json(
            partial_path,
            {"complete": False, "completed_results": results},
        )
        print(
            f"RESULT {model} accuracy={result['accuracy']:.3%} "
            f"latency_ms={result['average_latency_ms']:.1f}",
            flush=True,
        )

    nsos = load_nsos_results(
        args.nsos_report.resolve(), dataset, full_holdout
    )
    scoreboard = [
        {
            "model": result["model"],
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
            "adaptation_scope": "pretrained_q4_k_m_zero_shot",
        }
        for result in results
    ] + nsos["results"]
    report = {
        "schema_version": 1,
        "complete": True,
        "generated_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "purpose": "real_public_models_same_nsos_babi_holdout",
        "isolation": {
            "nsos_product_dependency": False,
            "public_model_runtime": "external Ollama HTTP API",
            "pytorch_used": False,
        },
        "dataset": {
            "name": "facebook/babi_qa, en-10k-qa1",
            "revision": BABI_REVISION,
            "hashes": dataset["hashes"],
            "train_examples": len(dataset["train"]),
            "test_examples": len(dataset["test"]),
            "evaluated_examples": len(cases),
            "case_manifest_sha256": canonical_sha256(
                [
                    {
                        "test_index": case["test_index"],
                        "sha256": case["sha256"],
                    }
                    for case in cases
                ]
            ),
        },
        "protocol": {
            "task": "single-word answer from provided bAbI facts",
            "temperature": 0.0,
            "seed": 20260729,
            "top_k": 1,
            "num_predict": 16,
            "num_ctx": 1024,
            "scoring": (
                "exact normalized JSON answer field; no LLM judge"
            ),
            "output_space": {
                "classes": len(output_vocabulary),
                "vocabulary": sorted(output_vocabulary),
                "vocabulary_sha256": canonical_sha256(
                    sorted(output_vocabulary)
                ),
                "constraint": (
                    "Ollama JSON schema enum over the complete NSOS "
                    "train vocabulary"
                ),
            },
            "all_holdout_examples": full_holdout,
            "determinism_replay_samples": min(
                args.determinism_samples, len(cases)
            ),
        },
        "comparison_scope": {
            "shared": [
                "identical immutable bAbI QA1 test file and SHA-256",
                (
                    f"identical {len(cases):,} held-out contexts, questions "
                    "and targets"
                ),
                "single-token answer objective",
                "deterministic decoding and exact scoring without LLM judge",
            ],
            "material_difference": (
                "NSOS arms were trained end to end from random "
                "initialization for 1,500 updates; public checkpoints are "
                "pretrained and evaluated zero-shot in Q4_K_M. The "
                "scoreboard compares held-out task behavior, not training "
                "compute or pretraining-data equivalence."
            ),
        },
        "ollama": {
            "api": args.api,
            "version": get_json(
                f"{args.api}/api/version", args.timeout
            ),
        },
        "git": provenance,
        "public_models": results,
        "nsos": nsos,
        "scoreboard": scoreboard,
        "accuracy_summary": {
            result["model"]: result["accuracy"] for result in scoreboard
        },
        "public_accuracy_mean": statistics.fmean(
            result["accuracy"] for result in results
        ),
        "cases": cases,
    }
    atomic_write_json(report_path, report)
    print(f"REPORT {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
