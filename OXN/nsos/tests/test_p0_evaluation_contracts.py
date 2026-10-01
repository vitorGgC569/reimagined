"""Offline numerical/orchestration regressions: no checkpoint or network."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from eval.adapters.dummy_adapter import DummyAdapter
from eval.benchmarks import ptbr_perplexity, wikitext
from eval.benchmarks.base import BenchmarkResult, measurement_errors
from eval.scoring.perplexity import sliding_window_perplexity, token_perplexity
from eval.orchestrator import run_scorecard
import run_scorecard as cli


class EvaluationContracts(unittest.TestCase):
    def test_uniform_distribution_and_window_accounting(self):
        for vocab in (2, 17, 256):
            for length in (2, 7, 8, 9, 15, 16, 17, 31):
                for stride in (1, 3, 7):
                    adapter = DummyAdapter(vocab_size=vocab, max_seq_len=8)
                    stats = sliding_window_perplexity(adapter, "x" * length, stride=stride)
                    self.assertAlmostEqual(stats["ppl"], vocab)
                    self.assertEqual(stats["n_tokens"], length - 1)
        self.assertAlmostEqual(token_perplexity(DummyAdapter(), "abc")["ppl"], 256)

    def test_real_ptbr_runner_uses_ppl_not_missing_key(self):
        with mock.patch.object(ptbr_perplexity, "_load_text", return_value="texto de teste " * 3):
            result = ptbr_perplexity.run(DummyAdapter(), n_documents=9)
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.metrics["perplexity"], 256)
        self.assertEqual(result.n_examples, result.metrics["n_tokens"])
        self.assertFalse(measurement_errors(result))
        self.assertEqual(len(result.extra["text_sha256"]), 64)

    def test_missing_stats_and_impossible_measurements_fail(self):
        for bad in ({}, {"ppl": 0, "n_tokens": 2, "nll_sum": 0},
                    {"ppl": 1, "n_tokens": 0, "nll_sum": 0},
                    {"ppl": 2, "n_tokens": 2, "nll_sum": 0},
                    {"ppl": float("nan"), "n_tokens": 2, "nll_sum": 1}):
            with mock.patch.object(ptbr_perplexity, "_load_text", return_value="abc"), \
                 mock.patch.object(ptbr_perplexity, "sliding_window_perplexity", return_value=bad):
                result = ptbr_perplexity.run(DummyAdapter())
            self.assertEqual(result.status, "error")
            self.assertTrue(measurement_errors(result))

    def test_invalid_scoring_limits_and_probabilities(self):
        adapter = DummyAdapter(max_seq_len=8)
        for options in ({"stride": 0}, {"stride": -1}, {"stride": 8},
                        {"max_tokens": 0}, {"max_tokens": 1}):
            with self.assertRaises(ValueError):
                sliding_window_perplexity(adapter, "abcdefghij", **options)
        for text in ("", "a"):
            with self.assertRaises(ValueError):
                sliding_window_perplexity(adapter, text)
        for value in (float("nan"), float("inf"), 0.1, True):
            with mock.patch.object(adapter, "score_tokens", return_value=value), self.assertRaises(ValueError):
                sliding_window_perplexity(adapter, "abc")
        with self.assertRaises(ValueError):
            sliding_window_perplexity(DummyAdapter(max_seq_len=1), "abc")

    def test_orchestrator_rejects_zero_ppl_and_preserves_kind(self):
        def bad_runner(adapter):
            return BenchmarkResult("bad", "perplexity", {"perplexity": 0.0}, 20, 0.1)
        result = run_scorecard(DummyAdapter(), benchmarks=[("bad", bad_runner, {}, False)])
        self.assertEqual(result.benchmarks[0].status, "error")
        self.assertTrue(cli.scorecard_failures(result))
        self.assertEqual(result.measurement_kind, "synthetic_smoke")
        for args in (["--adapter", "dummy", "--copy-to-docs", "--full"],
                     ["--adapter", "nsos", "--copy-to-docs"]):
            with self.assertRaises(SystemExit) as caught:
                cli.main(args)
            self.assertEqual(caught.exception.code, 2)

    def test_input_identity_and_ptbr_wikitext_agreement(self):
        digests = []
        with tempfile.TemporaryDirectory() as directory:
            for text in ("um texto repetido" * 3, "um texto repetido" * 3, "outro texto" * 3):
                with mock.patch.object(ptbr_perplexity, "_load_text", return_value=text), \
                     mock.patch.object(wikitext, "_load_text", return_value=text):
                    result = run_scorecard(DummyAdapter(), out_dir=directory, benchmarks=[
                        ("ptbr_perplexity", ptbr_perplexity.run, {}, False),
                        ("wikitext2_ppl", wikitext.run, {}, False)])
                self.assertFalse(cli.scorecard_failures(result))
                self.assertAlmostEqual(result.benchmarks[0].metrics["perplexity"],
                                       result.benchmarks[1].metrics["ppl"])
                digests.append(result.benchmarks[0].extra["evaluation_inputs"]["sha256"])
                data = json.loads((Path(directory) / result.timestamp / "scorecard.json").read_text())
                self.assertEqual(data["measurement_kind"], "synthetic_smoke")
                self.assertEqual(len(data["evidence_identity"]["eval_code_sha256"]), 64)
        self.assertEqual(digests[0], digests[1])
        self.assertNotEqual(digests[0], digests[2])


if __name__ == "__main__":
    unittest.main()
