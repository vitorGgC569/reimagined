"""Independent integrity and safety gate for a filtered Canarim SFT export.

This verifier is intentionally separate from the filtering pass.  It checks
that the published JSONL files match their manifest, that records are
reproducible, and that no accepted record contains a hard-reject pattern.
It never modifies the dataset; it writes only ``validation_report.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pyarrow.parquet as pq


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_filter_module() -> Any:
    path = Path(__file__).with_name("filter_canarim_sft.py")
    spec = importlib.util.spec_from_file_location("filter_canarim_sft", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import filter module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], str]:
    rows: List[Dict[str, Any]] = []
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            if not raw.strip():
                raise ValueError(f"blank line in {path.name}")
            rows.append(json.loads(raw.decode("utf-8")))
    return rows, digest.hexdigest()


def inspect_records(rows: Iterable[Dict[str, Any]], split: str, module: Any) -> Dict[str, Any]:
    errors: List[str] = []
    prompts: set[str] = set()
    records: set[str] = set()
    pattern_hits: Dict[str, int] = {}
    url_re = re.compile(r"(?:https?://|www\.)\S+", re.I)
    email_re = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
    for index, row in enumerate(rows):
        if set(row) != {"answer", "kind", "metadata", "prompt"}:
            errors.append(f"{split}[{index}]:unexpected_keys")
        if row.get("kind") != "instruction" or not isinstance(row.get("prompt"), str) or not row.get("prompt"):
            errors.append(f"{split}[{index}]:invalid_record_shape")
        if not isinstance(row.get("answer"), str) or not row.get("answer"):
            errors.append(f"{split}[{index}]:empty_answer")
        metadata = row.get("metadata") or {}
        prompt = row.get("prompt", "")
        answer = row.get("answer", "")
        prompt_hash = stable_hash(prompt)
        record_hash = stable_hash(prompt + "\0" + answer)
        if metadata.get("prompt_sha256") != prompt_hash:
            errors.append(f"{split}[{index}]:prompt_hash_mismatch")
        if metadata.get("record_sha256") != record_hash:
            errors.append(f"{split}[{index}]:record_hash_mismatch")
        if prompt_hash in prompts:
            errors.append(f"{split}[{index}]:duplicate_prompt")
        if record_hash in records:
            errors.append(f"{split}[{index}]:duplicate_record")
        prompts.add(prompt_hash)
        records.add(record_hash)
        combined = prompt + "\n" + answer
        for name, regex in (
            ("replacement", module.REPLACEMENT_RE),
            ("html", module.HTML_RE),
            ("control", module.CONTROL_RE),
            ("out_of_scope", module.OUT_OF_SCOPE_RE),
            ("unsafe", module.UNSAFE_OR_SENSITIVE_RE),
            ("medical", module.MEDICAL_RE),
            ("health_topic", module.HEALTH_TOPIC_RE),
            ("financial_topic", module.FINANCIAL_TOPIC_RE),
            ("agent_command_topic", module.AGENT_COMMAND_RE),
            ("math_topic", module.MATH_TOPIC_RE),
            ("web_content_or_news_task", module.WEB_CONTENT_RE),
            ("english_content", module.ENGLISH_CONTENT_RE),
            ("benchmark_task_meta", module.TASK_META_RE),
            ("sentiment_rewrite_task", module.SENTIMENT_RE),
            ("grammar_diagnosis_task", module.GRAMMAR_TASK_RE),
            ("template_placeholder", module.PLACEHOLDER_RE),
            ("imperial_localization", module.IMPERIAL_RE),
            ("pt_pt", module.PT_PT_RE),
            ("other_language_request", module.OTHER_LANGUAGE_REQUEST_RE),
        ):
            if regex.search(combined):
                pattern_hits[name] = pattern_hits.get(name, 0) + 1
        if module.INCOMPLETE_PROMPT_RE.search(prompt):
            pattern_hits["incomplete_prompt"] = pattern_hits.get("incomplete_prompt", 0) + 1
        if module.ENTITY_ANCHOR_RE.search(prompt) and not module.ENTITY_ANCHOR_RE.search(answer):
            pattern_hits["entity_anchor_mismatch"] = pattern_hits.get("entity_anchor_mismatch", 0) + 1
        if url_re.search(combined):
            pattern_hits["url"] = pattern_hits.get("url", 0) + 1
        if email_re.search(combined):
            pattern_hits["email"] = pattern_hits.get("email", 0) + 1
    return {"rows": len(prompts), "errors": errors, "pattern_hits": pattern_hits, "prompts": prompts}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.filtered_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    module = load_filter_module()
    report: Dict[str, Any] = {"filter_version": manifest.get("filter_version"), "errors": [], "warnings": []}

    for split in ("train", "eval"):
        raw_prefix = "train" if split == "train" else "test"
        raw_path = next((args.raw_dir / "data").glob(f"{raw_prefix}-*.parquet"))
        expected_source = manifest["counts"][f"source_{split}"]
        actual_source = pq.ParquetFile(raw_path).metadata.num_rows
        if actual_source != expected_source:
            report["errors"].append(f"source_{split}_row_count:{actual_source}!={expected_source}")

    split_results: Dict[str, Any] = {}
    for name in ("train", "eval", "quarantine_train", "quarantine_eval", "rejected_train", "rejected_eval"):
        path = args.filtered_dir / f"{name}.jsonl"
        if not path.exists():
            report["errors"].append(f"missing_file:{name}.jsonl")
            continue
        rows, digest = read_jsonl(path)
        expected = manifest.get("files", {}).get(f"{name}.jsonl", {})
        if expected.get("sha256") != digest:
            report["errors"].append(f"hash_mismatch:{name}.jsonl")
        if expected.get("rows") != len(rows):
            report["errors"].append(f"row_count_mismatch:{name}.jsonl")
        if name in ("train", "eval"):
            split_results[name] = inspect_records(rows, name, module)

    train_prompts = split_results.get("train", {}).get("prompts", set())
    eval_prompts = split_results.get("eval", {}).get("prompts", set())
    overlap = train_prompts & eval_prompts
    if overlap:
        report["errors"].append(f"train_eval_prompt_overlap:{len(overlap)}")
    for split, result in split_results.items():
        if result["errors"]:
            report["errors"].extend(result["errors"][:50])
        for name, count in result["pattern_hits"].items():
            if count:
                report["errors"].append(f"{split}_forbidden_{name}:{count}")
    report["counts"] = {
        "train": split_results.get("train", {}).get("rows", 0),
        "eval": split_results.get("eval", {}).get("rows", 0),
        "train_eval_prompt_overlap": len(overlap),
    }
    report["pattern_hits"] = {split: result["pattern_hits"] for split, result in split_results.items()}
    report["manifest_ready_for_training"] = bool(manifest.get("ready_for_training"))
    report["passed"] = not report["errors"] and report["manifest_ready_for_training"]
    (args.filtered_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=list))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
