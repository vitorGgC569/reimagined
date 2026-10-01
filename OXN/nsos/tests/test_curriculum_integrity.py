import hashlib
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


NSOS_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = NSOS_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import nsos_curriculum_lib as curriculum
import train_curriculum as training


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CurriculumIntegrityTests(unittest.TestCase):
    def test_release_gate_rejects_missing_nonfinite_and_boolean_metrics(self) -> None:
        baseline = {
            phase: {key[:-4]: threshold for key, threshold in thresholds.items()}
            for phase, thresholds in training.RELEASE_GATE_THRESHOLDS.items()
        }
        self.assertTrue(training.evaluate_release_gate(baseline)["passed"])
        for phase, thresholds in training.RELEASE_GATE_THRESHOLDS.items():
            for key in thresholds:
                for bad in (None, float("nan"), float("inf"), -float("inf"), True, "bad"):
                    with self.subTest(phase=phase, metric=key, value=bad):
                        metrics = {p: dict(values) for p, values in baseline.items()}
                        metrics[phase][key[:-4]] = bad
                        gate = training.evaluate_release_gate(metrics)
                        self.assertFalse(gate["passed"])
                        json.dumps(gate, allow_nan=False)
                metrics = {p: dict(values) for p, values in baseline.items()}
                del metrics[phase][key[:-4]]
                self.assertFalse(training.evaluate_release_gate(metrics)["passed"])

    def test_repetition_scale_override_is_finite_and_fail_closed(self) -> None:
        profile = {"repetition_unlikelihood_scale": 0.25}
        with mock.patch.dict(os.environ, {"NSOS_RUL_SCALE": "0.5"}):
            self.assertEqual(
                training.phase_repetition_scale(profile, "phase"), 0.5
            )
        for invalid in ("not-a-number", "nan", "inf", "-inf"):
            with self.subTest(value=invalid):
                with mock.patch.dict(
                    os.environ, {"NSOS_RUL_SCALE": invalid}
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "must be a finite"
                    ):
                        training.phase_repetition_scale(profile, "phase")

    def test_all_profiles_have_valid_layer_schedules(self) -> None:
        for profile_name, profile in training.PROFILES.items():
            config = profile.get("model_config", {})
            for family in ("attention", "moe", "ttt"):
                period = int(config.get(f"{family}_period", 1))
                slot = int(config.get(f"{family}_slot", 0))
                with self.subTest(profile=profile_name, family=family):
                    self.assertGreater(period, 0)
                    self.assertGreaterEqual(slot, 0)
                    self.assertLess(slot, period)

    def test_semantic_identity_ignores_metadata(self) -> None:
        left = {"id": "a", "source": "one", "prompt": "p", "answer": "a"}
        right = {"id": "b", "source": "two", "prompt": "p", "answer": "a"}
        self.assertEqual(curriculum._row_split_key(left), curriculum._row_split_key(right))

    def test_expand_records_never_manufactures_duplicates(self) -> None:
        rows = [
            {"id": "a", "source": "one", "prompt": "p", "answer": "a"},
            {"id": "b", "source": "two", "prompt": "p", "answer": "a"},
            {"id": "c", "source": "three", "prompt": "q", "answer": "b"},
        ]
        expanded = curriculum._expand_records(rows, 50, random.Random(7))
        self.assertEqual(len(expanded), 2)
        self.assertEqual(
            len({curriculum._row_split_key(row) for row in expanded}),
            len(expanded),
        )

    def test_tokenizer_uses_train_split_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            _write_jsonl(
                train_path,
                [{"text": "TRAIN_ONLY_TOKEN natural training sentence"}],
            )
            _write_jsonl(
                eval_path,
                [{"text": "EVAL_SECRET_MUST_NOT_ENTER_TOKENIZER"}],
            )
            manifest = {
                "phases": [
                    {
                        "name": "phase3_curated_text",
                        "train_file": train_path.name,
                        "eval_file": eval_path.name,
                    }
                ]
            }
            (root / "curriculum_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            captured: list[str] = []

            def capture_texts(texts, target_vocab):
                del target_vocab
                captured.extend(texts)
                return []

            with mock.patch.object(curriculum, "learn_bpe_merges", capture_texts):
                curriculum.build_tokenizer_bundle(root, target_vocab=259)

            joined = "\n".join(captured)
            self.assertIn("TRAIN_ONLY_TOKEN", joined)
            self.assertNotIn("EVAL_SECRET_MUST_NOT_ENTER_TOKENIZER", joined)

    def test_manifest_detects_tampering_and_train_eval_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            phases = []
            for index, phase_name in enumerate(curriculum.PHASE_ORDER):
                train_path = root / f"{phase_name}.train.jsonl"
                eval_path = root / f"{phase_name}.eval.jsonl"
                train_rows = [{"prompt": f"train-{index}", "answer": "ok"}]
                eval_rows = [{"prompt": f"eval-{index}", "answer": "ok"}]
                phases.append(
                    {
                        "name": phase_name,
                        "train_file": train_path.name,
                        "eval_file": eval_path.name,
                        "train_samples": 1,
                        "eval_samples": 1,
                        "train_sha256": _write_jsonl(train_path, train_rows),
                        "eval_sha256": _write_jsonl(eval_path, eval_rows),
                    }
                )
            manifest_path = root / "curriculum_manifest.json"
            manifest_path.write_text(
                json.dumps({"phases": phases}), encoding="utf-8"
            )
            training.validate_curriculum_manifest(root, manifest_path)

            first_train = root / phases[0]["train_file"]
            first_train.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                training.validate_curriculum_manifest(root, manifest_path)

            leaked = [{"prompt": "same-content", "answer": "same-answer"}]
            for split in ("train", "eval"):
                key = f"{split}_file"
                path = root / phases[0][key]
                phases[0][f"{split}_sha256"] = _write_jsonl(path, leaked)
                phases[0][f"{split}_samples"] = 1
            manifest_path.write_text(
                json.dumps({"phases": phases}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "train/eval leakage"):
                training.validate_curriculum_manifest(root, manifest_path)


if __name__ == "__main__":
    unittest.main()
