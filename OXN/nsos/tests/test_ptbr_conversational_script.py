from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "train_ptbr_conversational.py"
)
SPEC = importlib.util.spec_from_file_location("train_ptbr_conversational", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _FakeSnapshot:
    def __init__(self, step: int, fail: bool = False) -> None:
        self.global_step = step
        self.fail = fail

    def write(self, model_path: str, state_path: str) -> None:
        if self.fail:
            raise RuntimeError("injected snapshot write failure")
        Path(model_path).write_bytes(b"model-snapshot")
        Path(state_path).write_bytes(b"trainer-snapshot")


class _FakeTrainer:
    def __init__(self, step: int, fail: bool = False) -> None:
        self.global_step_count = step
        self.fail = fail

    def capture_checkpoint_snapshot(self):
        return _FakeSnapshot(self.global_step_count, self.fail)

    def load_training_state(self, state_path, model_path, allow_legacy=False):
        del state_path, model_path, allow_legacy


class _FakeModel:
    def load(self, path: str, strict: bool) -> None:
        del path, strict


class _FakeTelemetryModel:
    def __init__(self, **values) -> None:
        self.values = values

    def runtime_telemetry(self):
        return dict(self.values)


class _FakeValidationTensor:
    def __init__(self, values) -> None:
        self.device = "cpu"
        self._values = values

    def numpy(self):
        return self._values


class _FakeValidationModel:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.forward_calls = 0
        self.reset_calls = 0
        self._training_mode = True

    def training_mode(self) -> bool:
        return self._training_mode

    def set_training_mode(self, value: bool) -> None:
        self._training_mode = bool(value)

    def reset_session(self) -> None:
        self.reset_calls += 1

    def forward_ids(self, inputs, state):
        del state
        self.forward_calls += 1
        values = MODULE.np.zeros(
            (len(inputs), self.vocab_size), dtype=MODULE.np.float32
        )
        return _FakeValidationTensor(values)


class PTBRConversationalScriptTests(unittest.TestCase):
    @staticmethod
    def _build_prepared_fixture_workspace(root: Path) -> dict:
        preset_name = "smoke"
        preset = dict(MODULE.PRESETS[preset_name])
        seed = 20260729
        license_policy = "commercial-strict"
        corpus_dir = root / "corpus"
        store = MODULE.CorpusStore(corpus_dir / "corpus.sqlite3")
        try:
            MODULE.insert_fixture_corpus(store, preset)
            revisions = {
                "fixture": MODULE.stable_hash_text("nsos-ptbr-fixture-v1")
            }
            corpus_manifest = MODULE.corpus_manifest(
                store,
                preset_name,
                preset,
                seed,
                True,
                license_policy,
                revisions,
            )
            corpus_manifest_path = corpus_dir / "corpus_manifest.json"
            MODULE.atomic_write_json(corpus_manifest_path, corpus_manifest)

            tokenizer_path = root / "tokenizer.nsos"
            tokenizer_path.write_bytes(b"fixture-tokenizer")
            phase_contracts = {
                "base_train": ("base", "train", preset["base_tokens"]),
                "base_eval": ("base", "eval", None),
                "continuation_train": (
                    "continuation",
                    "train",
                    preset["continuation_tokens"],
                ),
                "continuation_eval": ("continuation", "eval", None),
                "sft_train": ("sft", "train", preset["sft_unique_tokens"]),
                "sft_eval": ("sft", "eval", None),
            }
            phases = {}
            for key, (phase_name, split, target) in phase_contracts.items():
                phase_dir = root / "packs" / key
                phase_dir.mkdir(parents=True, exist_ok=True)
                shard_path = phase_dir / "fixture.bin"
                shard_payload = f"{key}-fixture".encode("ascii")
                shard_path.write_bytes(shard_payload)
                phases[key] = {
                    "phase": phase_name,
                    "split": split,
                    "directory": key,
                    "target_tokens": target,
                    "shards": [
                        {
                            "file": shard_path.name,
                            "bytes": len(shard_payload),
                            "sha256": MODULE.sha256_file(shard_path),
                        }
                    ],
                }
            pack_manifest_path = root / "packs" / "pack_manifest.json"
            dataset = MODULE.canonical_corpus_manifest(store, "base")
            MODULE.atomic_write_json(
                pack_manifest_path,
                {
                    "format_version": MODULE.PACK_FORMAT_VERSION,
                    "created_at": MODULE.utc_now(),
                    "preset": preset_name,
                    "dataset": {
                        "manifest_content_sha256": dataset["manifest_content_sha256"],
                        "selected_documents": dataset["selected_documents"],
                        "selected_tokens_estimated": dataset["selected_tokens"],
                        "filter_version": MODULE.FILTER_VERSION,
                        "dedup_version": MODULE.DEDUP_VERSION,
                        "quota_policy_version": MODULE.QUOTA_POLICY_VERSION,
                        "source_mix_policy": dict(MODULE.BASE_SOURCE_MIX),
                        "conditional_quality_max_share": MODULE.QUALITY_CONDITIONAL_MAX_SHARE,
                    },
                    "packing": {"version": MODULE.PACKING_VERSION},
                    "tokenizer": {
                        "path": tokenizer_path.name,
                        "sha256": MODULE.sha256_file(tokenizer_path),
                        "vocab_size": int(preset["target_vocab"]),
                        "special_tokens": MODULE.SPECIAL_TOKENS,
                        "eos_token_id": int(preset["target_vocab"]) - 1,
                    },
                    "corpus_manifest": {
                        "path": str(corpus_manifest_path.relative_to(root)),
                        "sha256": MODULE.sha256_file(corpus_manifest_path),
                    },
                    "phases": phases,
                },
            )
            legacy_identity = {
                "script_version": 1,
                "preset": preset_name,
                "preset_config": preset,
                "seed": seed,
                "fixture": True,
                "license_policy": license_policy,
            }
            store.set_meta("run_identity", MODULE.stable_json(legacy_identity))
        finally:
            store.close()
        return {
            "preset_name": preset_name,
            "preset": preset,
            "seed": seed,
            "license_policy": license_policy,
            "corpus_manifest": corpus_manifest_path,
            "pack_manifest": pack_manifest_path,
            "tokenizer": tokenizer_path,
        }

    def test_corpus_identity_accepts_only_exact_current_or_legacy_schema(self) -> None:
        preset = dict(MODULE.PRESETS["smoke"])
        expected = MODULE.corpus_recipe_identity(
            "smoke", preset, 7, True, "commercial-strict"
        )
        self.assertTrue(
            MODULE.corpus_identity_compatible(
                MODULE.stable_json(expected), expected
            )
        )
        legacy = {
            "script_version": 1,
            "preset": "smoke",
            "preset_config": preset,
            "seed": 7,
            "fixture": True,
            "license_policy": "commercial-strict",
        }
        self.assertTrue(
            MODULE.corpus_identity_compatible(
                MODULE.stable_json(legacy), expected
            )
        )
        for mutation in (
            {**legacy, "seed": 8},
            {**legacy, "unknown": "must-fail-closed"},
            {**legacy, "seed": True},
            {**legacy, "script_version": MODULE.SCRIPT_VERSION + 1},
        ):
            with self.subTest(mutation=mutation):
                self.assertFalse(
                    MODULE.corpus_identity_compatible(
                        MODULE.stable_json(mutation), expected
                    )
                )

    def test_prepare_migrates_legacy_identity_without_rewriting_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = self._build_prepared_fixture_workspace(root)
            immutable_paths = [
                fixture["corpus_manifest"],
                fixture["pack_manifest"],
                fixture["tokenizer"],
                *sorted((root / "packs").glob("*/*.bin")),
            ]
            before = {
                path: (MODULE.sha256_file(path), path.stat().st_mtime_ns)
                for path in immutable_paths
            }
            args = MODULE.argparse.Namespace(
                seed=fixture["seed"],
                fixture=True,
                license_policy=fixture["license_policy"],
                shuffle_buffer=16,
            )
            manifest_path = MODULE.prepare(
                args,
                fixture["preset_name"],
                fixture["preset"],
                root,
                nsos=None,
            )
            self.assertEqual(manifest_path, fixture["pack_manifest"])
            after = {
                path: (MODULE.sha256_file(path), path.stat().st_mtime_ns)
                for path in immutable_paths
            }
            self.assertEqual(before, after)

            store = MODULE.CorpusStore(root / "corpus" / "corpus.sqlite3")
            try:
                expected = MODULE.corpus_recipe_identity(
                    fixture["preset_name"],
                    fixture["preset"],
                    fixture["seed"],
                    True,
                    fixture["license_policy"],
                )
                self.assertEqual(
                    store.get_meta("run_identity"), MODULE.stable_json(expected)
                )
            finally:
                store.close()

    def test_workspace_validation_rejects_manifest_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = self._build_prepared_fixture_workspace(root)
            pack = MODULE._read_json_object(
                fixture["pack_manifest"], "Pack manifest"
            )
            pack["tokenizer"]["path"] = "../outside.nsos"
            MODULE.atomic_write_json(fixture["pack_manifest"], pack)
            with self.assertRaisesRegex(RuntimeError, "escapa do workspace"):
                MODULE.validate_training_workspace(
                    root,
                    fixture["preset_name"],
                    fixture["preset"],
                    fixture["seed"],
                    True,
                    fixture["license_policy"],
                )

    def test_commercial_policy_rejects_noncommercial_and_unknown(self) -> None:
        allowed, decision, _ = MODULE.source_allowed(
            "https://example/Corpus-Carolina", "carolina", "commercial-strict"
        )
        self.assertFalse(allowed)
        self.assertEqual(decision, "deny")
        allowed, decision, _ = MODULE.source_allowed(
            "https://example/sem-regra", "desconhecido", "commercial-strict"
        )
        self.assertFalse(allowed)
        self.assertEqual(decision, "unknown")

    def test_sft_examples_mask_only_assistant_spans(self) -> None:
        messages = [
            {"role": "system", "content": "Sistema"},
            {"role": "user", "content": "Pergunta um"},
            {"role": "assistant", "content": "Resposta um"},
            {"role": "user", "content": "Pergunta dois"},
            {"role": "assistant", "content": "Resposta dois"},
        ]
        examples = list(MODULE.supervised_examples(messages))
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0][1], "Resposta um")
        self.assertNotIn("Resposta um", examples[0][0])
        self.assertIn("Resposta um", examples[1][0])
        self.assertEqual(examples[1][1], "Resposta dois")

    def test_fixture_is_deterministic_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "corpus.sqlite3"
            store = MODULE.CorpusStore(path)
            try:
                MODULE.insert_fixture_corpus(store, MODULE.PRESETS["smoke"])
                first = {
                    phase: store.counts(phase).__dict__
                    for phase in ("base", "continuation", "sft")
                }
                MODULE.insert_fixture_corpus(store, MODULE.PRESETS["smoke"])
                second = {
                    phase: store.counts(phase).__dict__
                    for phase in ("base", "continuation", "sft")
                }
                self.assertEqual(first, second)
                overlap = store.connection.execute(
                    """
                    SELECT COUNT(*) FROM records a
                    JOIN records b ON a.fingerprint = b.fingerprint
                    WHERE a.phase != b.phase
                    """
                ).fetchone()[0]
                self.assertEqual(overlap, 0)
            finally:
                store.close()

    def test_chat_format_uses_explicit_roles(self) -> None:
        text = MODULE.format_chat(
            [
                {"role": "system", "content": "Sistema"},
                {"role": "user", "content": "Olá"},
                {"role": "assistant", "content": "Oi"},
            ]
        )
        self.assertTrue(text.startswith(MODULE.BOS_TOKEN))
        self.assertIn("<|system|>", text)
        self.assertIn("<|user|>", text)
        self.assertIn("<|assistant|>", text)
        self.assertTrue(text.endswith(MODULE.EOS_TOKEN))

    def test_causal_step_count_reserves_the_next_token(self) -> None:
        manifest = {
            "phases": {
                "base_train": {"shards": [{"tokens": 8_192}]},
                "continuation_train": {"shards": [{"tokens": 4_096}]},
                "sft_train": {"shards": []},
            }
        }
        preset = {
            "seq_len": 64,
            "causal_batch": 1,
            "sft_batch": 1,
            "sft_epochs": 1,
        }
        counts = MODULE.phase_step_counts(Path("."), manifest, preset)
        self.assertEqual(counts, {"base": 127, "continuation": 63, "sft": 0})

    def test_causal_validation_accepts_exact_single_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            shard_path = Path(temp_dir) / "eval.u16"
            tokens = MODULE.array("H", [0, 1, 2, 3, 0])
            if MODULE.sys.byteorder != "little":
                tokens.byteswap()
            shard_path.write_bytes(tokens.tobytes())

            class _FakeNsos:
                class Device:
                    GPU = "gpu"

            model = _FakeValidationModel(vocab_size=4)
            loss = MODULE.causal_validation_loss(
                _FakeNsos,
                model,
                shard_path,
                seq_len=4,
                max_windows=1,
            )

            self.assertTrue(MODULE.np.isfinite(loss))
            self.assertAlmostEqual(loss, MODULE.np.log(4.0))
            self.assertEqual(model.forward_calls, 1)
            self.assertEqual(model.reset_calls, 2)
            self.assertTrue(model.training_mode())

    def test_mamba_only_presets_do_not_enable_implicit_qat(self) -> None:
        for name, preset in MODULE.PRESETS.items():
            with self.subTest(preset=name):
                self.assertFalse(preset["progressive_qat"])

    def test_sft_estimate_counts_only_capped_assistant_targets(self) -> None:
        messages = [
            {"role": "system", "content": "S" * 2_000},
            {"role": "user", "content": "U" * 2_000},
            {"role": "assistant", "content": "A" * 10_000},
        ]
        self.assertEqual(
            MODULE.estimate_sft_target_tokens(messages, seq_len=512),
            256,
        )

    def test_sft_estimate_migration_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MODULE.CorpusStore(Path(temp_dir) / "corpus.sqlite3")
            try:
                messages = [
                    {"role": "system", "content": "Sistema"},
                    {"role": "user", "content": "Pergunta " * 100},
                    {"role": "assistant", "content": "Resposta curta"},
                ]
                store.insert(
                    fingerprint="fixture-sft-estimate",
                    phase="sft",
                    split="train",
                    category="general",
                    source="fixture",
                    subset="fixture",
                    payload=MODULE.stable_json(messages),
                    estimated_tokens=999,
                    quality=5.0,
                    license_decision="autoral",
                )
                store.commit()
                MODULE.migrate_sft_estimates_to_supervised_targets(store, 512)
                first = store.counts("sft").estimated_tokens
                MODULE.migrate_sft_estimates_to_supervised_targets(store, 512)
                second = store.counts("sft").estimated_tokens
                self.assertLess(first, 999)
                self.assertEqual(first, second)
            finally:
                store.close()

    def test_mmap_shards_preserve_order_slices_and_explicit_close(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            token_writer = MODULE.TokenShardWriter(root, "tokens", 64)
            expected_tokens = [0, 7, 7, 2, 65535, 11, 3]
            token_writer.add(expected_tokens)
            token_spec = token_writer.finish()[0]
            token_path = root / token_spec["file"]
            token_shard = MODULE.read_u16_tokens(token_path)
            with token_shard as shard:
                self.assertEqual(len(shard), len(expected_tokens))
                self.assertEqual(shard[:], expected_tokens)
                self.assertEqual(shard[1:6:2], expected_tokens[1:6:2])
                self.assertEqual(shard[-1], expected_tokens[-1])
            with self.assertRaises(ValueError):
                _ = token_shard[0]
            renamed = root / "tokens-closed.u16"
            token_path.rename(renamed)

            sft_writer = MODULE.SFTShardWriter(root, records_per_shard=8)
            records = [([1, 2], [3]), ([4], [5, 6]), ([7, 8, 9], [10, 11])]
            for prompt, answer in records:
                sft_writer.add(prompt, answer)
            sft_spec = sft_writer.finish()[0]
            sft_path = root / sft_spec["file"]
            sft_shard = MODULE.read_sft_records(sft_path)
            with sft_shard as shard:
                self.assertEqual(len(shard), len(records))
                self.assertEqual(shard[:], records)
                self.assertEqual(shard[-1], records[-1])
            with self.assertRaises(ValueError):
                _ = sft_shard[0]
            sft_path.rename(root / "sft-closed.bin")

    def test_mmap_sft_rejects_trailing_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.bin"
            path.write_bytes(
                MODULE.SFTShardWriter.MAGIC
                + MODULE.struct.pack("<II", MODULE.PACK_FORMAT_VERSION, 0)
                + b"x"
            )
            with self.assertRaisesRegex(RuntimeError, "Bytes extras"):
                MODULE.read_sft_records(path)

    def test_sft_token_telemetry_matches_native_bucket_contract(self) -> None:
        prompts = [[1] * 10, [2] * 7, [3] * 2]
        answers = [[4] * 4, [5] * 3, [6] * 2]
        counts = MODULE.sft_step_token_counts(prompts, answers)
        self.assertEqual(counts["non_padding_tokens"], 13 + 9 + 3)
        self.assertEqual(counts["supervised_tokens"], 9)
        self.assertEqual(counts["assistant_tokens"], 9)
        self.assertGreaterEqual(
            counts["raw_tokens"], counts["non_padding_tokens"]
        )

    def test_gpu_runtime_audit_rejects_every_host_fallback_class(self) -> None:
        for field in (
            "mamba_fast_path_fallbacks",
            "faithful_forward_host_fallbacks",
            "faithful_backward_host_fallbacks",
            "faithful_streaming_host_fallbacks",
            "stream_priming_host_fallbacks",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(RuntimeError, field):
                    MODULE.audit_deterministic_gpu_runtime(
                        _FakeTelemetryModel(**{field: 1})
                    )

    def test_rolling_telemetry_uses_window_and_first_1000_percentiles(self) -> None:
        telemetry = MODULE.RollingTrainingTelemetry(window_steps=100)
        native = {
            "enabled": False,
            "global_step": 0,
            "bucket_count": 0,
            **{key: 0.0 for key in MODULE.NATIVE_TIMING_KEYS},
        }
        for step in range(1, 1_101):
            telemetry.observe(
                phase="base" if step <= 1_000 else "continuation",
                global_step=step,
                elapsed_seconds=0.1,
                loss=5.0 - step * 0.001,
                token_counts={
                    "raw_tokens": 200,
                    "non_padding_tokens": 180,
                    "supervised_tokens": 160,
                    "assistant_tokens": 0,
                },
                native_timing=native,
            )
        snapshot = telemetry.snapshot()
        self.assertEqual(snapshot["rolling"]["steps"], 100)
        self.assertEqual(snapshot["run"]["steps"], 1_100)
        self.assertEqual(snapshot["phases"]["base"]["steps"], 1_000)
        self.assertEqual(len(telemetry._benchmark_samples), 1_000)
        self.assertFalse(hasattr(telemetry, "_all_samples"))
        benchmark = telemetry.benchmark_percentiles()
        self.assertEqual(benchmark["observed_steps"], 1_000)
        self.assertEqual(benchmark["measured_steps"], 900)
        raw = benchmark["percentiles"]["raw_tokens_per_second"]
        self.assertIsNotNone(raw)
        self.assertAlmostEqual(raw["p50"], 2_000.0)
        trace = telemetry.determinism_trace()
        self.assertEqual(len(trace), 1_000)
        self.assertEqual(trace[0]["run_step"], 1)
        self.assertEqual(trace[-1]["global_step"], 1_000)
        self.assertEqual(trace[0]["loss_hex"], float(4.999).hex())
        self.assertNotIn("elapsed_seconds", trace[0])

    def test_async_checkpoint_commits_hashes_and_rejects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tokenizer = root / "tokenizer.nsos"
            tokenizer.write_bytes(b"tokenizer")
            manager = MODULE.CheckpointManager(
                root / "run",
                _FakeModel(),
                _FakeTrainer(37),
                tokenizer,
                {"fixture": "async-checkpoint"},
                keep_periodic=2,
            )
            try:
                generation = manager.save(
                    {"phase": "base", "offset": 9},
                    kind="periodic",
                    label="fixture",
                    latest_loss=1.25,
                )
                verification = manager.verify_all()
                self.assertEqual(verification["valid"], 1)
                self.assertEqual(verification["invalid"], 0)
                self.assertEqual(
                    verification["latest_file_sha256"]["model.bin"],
                    MODULE.sha256_file(generation / "model.bin"),
                )
                self.assertEqual(
                    verification["latest_file_sha256"]["trainer.state"],
                    MODULE.sha256_file(generation / "trainer.state"),
                )
                self.assertEqual(
                    verification["latest_identity_sha256"],
                    manager.identity_sha256,
                )
                self.assertTrue((generation / "COMMITTED").is_file())
                manifest = MODULE.json.loads(
                    (generation / "checkpoint_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(manifest["global_step"], 37)
                self.assertEqual(
                    manifest["format_version"],
                    MODULE.CHECKPOINT_FORMAT_VERSION,
                )
                self.assertIn("checkpoint_timing", manifest)
                committed = MODULE.json.loads(
                    (generation / "COMMITTED").read_text(encoding="ascii")
                )
                self.assertEqual(
                    committed["manifest_sha256"],
                    MODULE.sha256_file(
                        generation / "checkpoint_manifest.json"
                    ),
                )
                timing = manager.timing_summary()
                self.assertEqual(timing["completed"], 1)
                self.assertFalse(timing["pending"])

                manifest_path = generation / "checkpoint_manifest.json"
                original_manifest = manifest_path.read_bytes()
                manifest["global_step"] = 999_999
                MODULE.atomic_write_json(manifest_path, manifest)
                valid, _ = manager._manifest_valid(generation)
                self.assertFalse(valid)
                manifest_path.write_bytes(original_manifest)
                valid, _ = manager._manifest_valid(generation)
                self.assertTrue(valid)

                (generation / "model.bin").write_bytes(b"corrupted")
                valid, _ = manager._manifest_valid(generation)
                self.assertFalse(valid)
            finally:
                manager.close()

    def test_atomic_json_uses_short_same_directory_temporary_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            # Reproduce the long descriptive generation paths used by the
            # production checkpoint manager. The final JSON path remains
            # usable while the former target-name/PID/thread/time temporary
            # basename crossed the traditional Win32 path limit.
            budget = max(210 - len(str(root)), 32)
            deep = root / ("g" * budget)
            deep.mkdir()
            target = deep / "checkpoint_manifest.json"
            MODULE.atomic_write_json(target, {"step": 10, "valid": True})
            self.assertEqual(
                MODULE.json.loads(target.read_text(encoding="utf-8")),
                {"step": 10, "valid": True},
            )
            self.assertEqual(
                [path for path in deep.iterdir() if path.name.startswith(".tmp-")],
                [],
            )

    def test_async_checkpoint_surfaces_worker_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tokenizer = root / "tokenizer.nsos"
            tokenizer.write_bytes(b"tokenizer")
            manager = MODULE.CheckpointManager(
                root / "run",
                _FakeModel(),
                _FakeTrainer(4, fail=True),
                tokenizer,
                {"fixture": "failure"},
                keep_periodic=1,
            )
            manager.save(
                {"phase": "base"},
                kind="periodic",
                label="failure",
                latest_loss=2.0,
            )
            with self.assertRaisesRegex(
                RuntimeError, "injected snapshot write failure"
            ):
                manager.close()

    def test_async_checkpoint_failure_poison_prevents_second_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tokenizer = root / "tokenizer.nsos"
            tokenizer.write_bytes(b"tokenizer")
            manager = MODULE.CheckpointManager(
                root / "run",
                _FakeModel(),
                _FakeTrainer(4, fail=True),
                tokenizer,
                {"fixture": "poisoned-checkpoint"},
                keep_periodic=1,
            )
            manager.save(
                {"phase": "base"},
                kind="periodic",
                label="first-failure",
                latest_loss=2.0,
            )
            with self.assertRaisesRegex(
                RuntimeError, "injected snapshot write failure"
            ):
                manager.verify_all()
            with self.assertRaisesRegex(
                RuntimeError, "falha assíncrona anterior"
            ):
                manager.save(
                    {"phase": "base"},
                    kind="milestone",
                    label="must-not-queue",
                    latest_loss=2.0,
                )
            self.assertIsNone(manager._pending_checkpoint)
            # The first failure was already surfaced; close only joins the
            # worker and must not mask the original exception.
            manager.close()

    def test_checkpoint_fallback_copy_publishes_committed_last(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tokenizer = root / "tokenizer.nsos"
            tokenizer.write_bytes(b"tokenizer")
            manager = MODULE.CheckpointManager(
                root / "run",
                _FakeModel(),
                _FakeTrainer(51),
                tokenizer,
                {"fixture": "rename-fallback"},
                keep_periodic=1,
            )
            try:
                with mock.patch.object(
                    MODULE.os,
                    "rename",
                    side_effect=PermissionError("injected rename denial"),
                ), mock.patch.object(MODULE.time, "sleep", return_value=None):
                    generation = manager.save(
                        {"phase": "base"},
                        kind="periodic",
                        label="fallback",
                        latest_loss=1.0,
                    )
                    verification = manager.verify_all()
                self.assertEqual(verification["valid"], 1)
                self.assertEqual(verification["invalid"], 0)
                self.assertTrue((generation / "COMMITTED").is_file())
                self.assertFalse(
                    (generation.parent / f".staging-{generation.name}").exists()
                )
            finally:
                manager.close()

    def test_corrupt_latest_pointer_is_reported_before_generation_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tokenizer = root / "tokenizer.nsos"
            tokenizer.write_bytes(b"tokenizer")
            manager = MODULE.CheckpointManager(
                root / "run",
                _FakeModel(),
                _FakeTrainer(73),
                tokenizer,
                {"fixture": "corrupt-latest-pointer"},
                keep_periodic=1,
            )
            try:
                expected = manager.save(
                    {"phase": "base"},
                    kind="periodic",
                    label="valid-generation",
                    latest_loss=1.0,
                )
                manager.verify_all()
                (manager.root / "LATEST.json").write_text(
                    "{invalid-json", encoding="utf-8"
                )
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    recovered = manager.latest()
                self.assertIsNotNone(recovered)
                assert recovered is not None
                self.assertEqual(recovered[0], expected)
                self.assertIn("LATEST.json inválido", stderr.getvalue())
            finally:
                manager.close()

    def test_legacy_identity_migration_is_explicit_and_shared_fields_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tokenizer = root / "tokenizer.nsos"
            tokenizer.write_bytes(b"tokenizer")
            current = {
                "script_version": MODULE.SCRIPT_VERSION,
                "script_sha256": "new-script",
                "preset": "pilot",
                "preset_config": {"seq_len": 128},
                "seed": 7,
                "model_config": {"d_model": 64},
                "pack_manifest_sha256": "pack",
                "tokenizer_sha256": "tokenizer",
                "build_dir": "build",
                "deterministic_reductions": True,
                "runtime_execution_identity": {"precision.matmul": "fp32"},
                "runtime_module_sha256": "new-module",
            }
            legacy = {
                key: value
                for key, value in current.items()
                if key not in (
                    "script_sha256",
                    "runtime_module_sha256",
                )
            }
            legacy["script_version"] = MODULE.SCRIPT_VERSION - 1

            strict = MODULE.CheckpointManager(
                root / "strict",
                _FakeModel(),
                _FakeTrainer(0),
                tokenizer,
                current,
                keep_periodic=1,
            )
            migrating = MODULE.CheckpointManager(
                root / "migrating",
                _FakeModel(),
                _FakeTrainer(0),
                tokenizer,
                current,
                keep_periodic=1,
                allow_legacy_runtime_identity=True,
            )
            try:
                self.assertFalse(strict._legacy_identity_compatible(legacy))
                self.assertTrue(migrating._legacy_identity_compatible(legacy))
                incompatible = dict(legacy)
                incompatible["deterministic_reductions"] = False
                self.assertFalse(
                    migrating._legacy_identity_compatible(incompatible)
                )
            finally:
                strict.close()
                migrating.close()


if __name__ == "__main__":
    unittest.main()
