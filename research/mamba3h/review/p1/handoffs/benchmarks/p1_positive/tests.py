"""Small executable positive-routing, anti-leakage and bounded-split tests."""

import copy
import os
from pathlib import Path
import unittest

from . import generate, build_corpus, solve, model_numeric, model_view
from .artifacts import p0_frozen
from .generation import _accept, _candidate, DuplicateBudgetExceeded, data_seed, input_fingerprint
from .oracle import UnboundQuery
from .routing import State, step, retrieve
from .schema import COUNTS, IGNORE, NAMESPACE, SEEDS, event, validate_profile, validate_model_view


def fixture(events):
    return {"task": "mqar", "group": "none", "events": events}


class P1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus, cls.sampling = build_corpus()

    def test_all_800_profiles_and_independent_solver(self):
        total = 0
        for (seed, split), episodes in self.corpus.items():
            self.assertEqual(len(episodes), COUNTS[split])
            for ep in episodes:
                validate_profile(ep)
                self.assertEqual(solve(ep), ep["targets"])
                self.assertEqual(ep["targets"][:5], [IGNORE] * 5)
                self.assertIn(ep["targets"][5], range(4))
                self.assertEqual(ep["query_mask"], [False] * 5 + [True])
                self.assertEqual(ep["metadata"]["namespace"], NAMESPACE)
                total += 1
        self.assertEqual(total, 800)

    def test_all_splits_seeds_globally_disjoint_inputs_and_data_seeds(self):
        inputs, seeds = set(), set()
        for episodes in self.corpus.values():
            for ep in episodes:
                fp, ds = input_fingerprint(ep), ep["metadata"]["data_seed"]
                self.assertNotIn(fp, inputs); inputs.add(fp)
                self.assertNotIn(ds, seeds); seeds.add(ds)
        self.assertEqual(len(inputs), 800)
        self.assertEqual(len(seeds), 800)

    def test_api_reproducible_and_prefix_independent_of_call_count(self):
        for split in COUNTS:
            self.assertEqual(generate("mqar", 11, split, 2), self.corpus[(11, split)][:2])
        self.assertEqual(generate("mqar", 23, "train", 0), [])
        self.assertEqual(generate("mqar", 71, "test", 1), generate("mqar", 71, "test", 4)[:1])

    def test_bounded_global_duplicate_rejection_and_exhaustion(self):
        first, second = _candidate(11, "train", 0, 0), _candidate(11, "train", 1, 0)
        seen = {input_fingerprint(first)}
        accepted, retries = _accept(lambda attempt: first if attempt == 0 else second, seen, 2)
        self.assertEqual(accepted, second)
        self.assertEqual(retries, 1)
        calls = []
        def repeated(attempt):
            calls.append(attempt)
            return first
        with self.assertRaises(DuplicateBudgetExceeded):
            _accept(repeated, seen, 3)
        self.assertEqual(calls, [0, 1, 2])

    def test_query_and_noop_have_only_id_and_identical_numeric_rows(self):
        ep = fixture([event("WRITE", 1, 3), event("NOOP", 1), event("QUERY", 1)])
        x = model_numeric(ep)
        self.assertEqual(x[0], [0., 1., 0., 0., 0., 0., 0., 1.])
        self.assertEqual(x[1], x[2])
        self.assertEqual(x[2][4:], [0.] * 4)
        self.assertTrue(all(len(row) == 8 and all(type(v) is float for v in row) for row in x))

    def test_all_positive_routing_exact_with_work_gates(self):
        for episodes in self.corpus.values():
            for ep in episodes:
                out, state, work = retrieve(ep)
                self.assertEqual(out[-1][4:], [float(j == ep["targets"][-1]) for j in range(4)])
                self.assertEqual(sum(w["write_operations"] for w in work), 3)
                self.assertEqual(sum(w["read_operations"] for w in work), 1)
                self.assertEqual(sum(w["similarity_comparisons"] for w in work), 3)
                self.assertTrue(all(w["similarity_comparisons"] == 0 for w in work[:5]))
                self.assertEqual(sum(slot is not None for slot in state.slots), 3)

    def test_poisoned_targets_metadata_do_not_change_input_or_retrieval(self):
        ep = copy.deepcopy(self.corpus[(11, "test")][0])
        expected = model_view(ep), model_numeric(ep), retrieve(ep), solve(ep)
        ep["targets"] = [999] * 6
        ep["metadata"] = {"future_answers": "poison"}
        ep["solver"] = "never forward to step"
        self.assertEqual((model_view(ep), model_numeric(ep), retrieve(ep), solve(ep)), expected)

    def test_no_future_write_can_answer_unbound_current_query(self):
        ep = fixture([event("QUERY", 2), event("WRITE", 2, 1)])
        with self.assertRaises(UnboundQuery):
            solve(ep)

    def test_future_suffix_cannot_change_past_query_or_routing(self):
        ep = fixture([event("WRITE", 2, 1), event("QUERY", 2), event("WRITE", 2, 3)])
        changed = copy.deepcopy(ep)
        changed["events"][-1]["value"] = 0
        self.assertEqual(solve(ep)[:2], solve(changed)[:2])
        self.assertEqual(model_numeric(ep)[:2], model_numeric(changed)[:2])
        self.assertEqual(retrieve(ep)[0][:2], retrieve(changed)[0][:2])

    def test_injected_query_value_event_target_and_control_target_rejected(self):
        for field in ("value", "answer"):
            ep = copy.deepcopy(self.corpus[(11, "train")][0])
            ep["events"][-1][field] = 2
            with self.assertRaises(ValueError):
                model_numeric(ep)
        view = model_view(self.corpus[(11, "train")][0])
        view["control"][-1]["target"] = 2
        with self.assertRaises(ValueError):
            validate_model_view(view)
        with self.assertRaises(ValueError):
            step(model_numeric(self.corpus[(11, "train")][0])[-1], State.empty(), control=view["control"][-1])

    def test_capacity_failure_never_silently_evicted(self):
        ep = fixture([event("WRITE", i, i) for i in range(4)])
        with self.assertRaises(ValueError):
            retrieve(ep)

    def test_independent_solver_does_not_import_generator_or_routing(self):
        source = Path(__file__).with_name("oracle.py").read_text("utf8")
        self.assertNotIn("from .generation", source)
        self.assertNotIn("from .routing", source)
        self.assertNotIn("targets\"]", source)

    def test_neutral_and_query_cannot_be_written_by_wrong_gate(self):
        ep = fixture([event("NOOP", 0)])
        control = model_view(ep)["control"][0]
        control["write"] = True
        with self.assertRaises(ValueError):
            step(model_numeric(ep)[0], State.empty(), control=control)

    def test_generator_rejects_unregistered_or_changed_protocol(self):
        for args, kwargs in ((("mqar", 12, "train", 1), {}), (("mqar", 11, "val", 1), {}),
            (("mqar", 11, "test", 65), {}), (("inst", 11, "test", 1), {}),
            (("mqar", 11, "test", 1), {"length": 7}), (("mqar", 11, "test", 1), {"operations": 3})):
            with self.assertRaises(ValueError):
                generate(*args, **kwargs)

    def test_namespace_and_sampling_budget_are_explicit(self):
        self.assertTrue(NAMESPACE.startswith("P1fresh"))
        self.assertEqual(self.sampling["accepted"], 800)
        self.assertLessEqual(self.sampling["attempts"], 800 * 16)
        self.assertNotEqual(data_seed(11, "train", 0), data_seed(11, "test", 0))
        self.assertNotEqual(data_seed(11, "train", 0, 0), data_seed(11, "train", 0, 1))

    def test_p0_all_pinned_files_still_frozen(self):
        status = p0_frozen()
        self.assertTrue(status["ok"], status)
        self.assertEqual(status["files_checked"], 79)

    def test_threads_at_most_two(self):
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.assertLessEqual(int(os.environ[name]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
