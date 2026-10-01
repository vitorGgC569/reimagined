"""Distribution/state-reference contracts; no trained checkpoint is required."""
from pathlib import Path
import math
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from speculative_decode import (apply_top_k_top_p, baseline_decode, get_logits,
                                sample_from, softmax_temperature, speculative_decode)


class PrefixModel:
    def __init__(self, wrong=False):
        self.wrong = wrong

    def next_token_logits(self, prefix):
        token = (sum(prefix) + len(prefix)) % 3
        if self.wrong:
            token = (token + 1) % 3
        return [4.0 if i == token else -4.0 for i in range(3)]


class DistributionModel:
    def __init__(self, probabilities):
        self.logits = [math.log(p) for p in probabilities]

    def next_token_logits(self, prefix):
        return self.logits


class SpeculativeReferenceTests(unittest.TestCase):
    def test_greedy_matches_baseline_with_agreeing_and_rejecting_draft(self):
        for wrong in (False, True):
            target, draft = PrefixModel(), PrefixModel(wrong)
            expected = baseline_decode(target, [1], -1, 17, 0.0, 0, 1.0, random.Random(4))
            actual, metrics = speculative_decode(target, draft, None, [1], -1,
                17, 4, 0.0, 0, 1.0, random.Random(4))
            self.assertEqual(actual, expected)
            self.assertLessEqual(metrics["tokens_accepted"], metrics["tokens_proposed"])

    def test_rejection_sampling_preserves_target_distribution(self):
        target = DistributionModel([.65, .25, .10])
        draft = DistributionModel([.10, .80, .10])
        rng = random.Random(1829)
        counts = [0, 0, 0]
        for _ in range(10000):
            actual, _ = speculative_decode(target, draft, None, [0], -1,
                1, 4, 1.0, 0, 1.0, rng)
            self.assertEqual(len(actual), 1)
            counts[actual[0]] += 1
        for count, probability in zip(counts, [.65, .25, .10]):
            self.assertAlmostEqual(count / 10000, probability, delta=.025)

    def test_greedy_fallback_is_rejected(self):
        class GreedyOnly:
            def next_token_greedy(self, history):
                return 0
        with self.assertRaises(TypeError):
            get_logits(GreedyOnly(), [0])

    def test_zero_mass_never_sampled_at_zero_uniform(self):
        class ZeroRandom:
            def random(self):
                return 0.0
        self.assertEqual(sample_from([0.0, 1.0, 0.0], ZeroRandom()), 1)

    def test_top_k_renormalizes_before_nucleus(self):
        self.assertEqual(apply_top_k_top_p([.40, .30, .20, .10], 2, .55), [1., 0., 0., 0.])

    def test_invalid_inputs_fail_closed(self):
        for values in ([], [math.nan], [math.inf]):
            with self.assertRaises(ValueError):
                softmax_temperature(values, 1.0)
        with self.assertRaises(ValueError):
            speculative_decode(PrefixModel(), PrefixModel(), None, [0], -1,
                1, 0, 1.0, 0, 1.0, random.Random(2))


if __name__ == "__main__":
    unittest.main()
