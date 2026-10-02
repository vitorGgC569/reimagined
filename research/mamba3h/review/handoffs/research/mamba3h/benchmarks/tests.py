"""Executable adversarial correctness, leakage, split and reporting tests."""

import copy
import itertools
import os
import random
import unittest
from collections import Counter
from pathlib import Path

from . import generate, solve, model_view, encode_numeric, compact_targets, label_vocabulary
from .controls import predict_first_write, predict_reversed_order, predict_undefined
from .generation import SEEDS, pair_partition, _online_labels
from .manifest import profiles
from .protocol import accuracy, paired_summary, gap_recovery, check_budget, factorial_summary, ARMS
from .schema import event, fingerprint, IGNORE, UNDEFINED, validate_model_view, CONTROL_FIELDS


def episode(task, events, group=None):
    return {"task": task, "group": group or ("none" if task == "mqar" else "s5"), "events": events}


class OracleTests(unittest.TestCase):
    def test_long_random_operation_chains(self):
        for task in ("group", "inst"):
            for group in ("s5", "abelian"):
                for ep in generate(task, 71, "test", 8, length=96, operations=64, queries=10, group=group):
                    self.assertEqual(solve(ep), ep["targets"])

    def test_mqar_overwrite_revoke_and_entity_isolation(self):
        ep = episode("mqar", [event("QUERY", 0), event("WRITE", 0, value=7), event("WRITE", 1, value=9),
             event("QUERY", 0), event("WRITE", 0, value=3), event("QUERY", 1), event("QUERY", 0),
             event("REVOKE", 0), event("QUERY", 0), event("QUERY", 1), event("WRITE", 0, value=4), event("QUERY", 0)])
        self.assertEqual([v for v in solve(ep) if v != IGNORE], [125, 7, 9, 3, 125, 9, 4])
        self.assertNotEqual(solve(ep), predict_first_write(ep))

    def test_inst_entity_order_and_overwrite(self):
        ep = episode("inst", [event("SET", 0, value=0), event("SET", 1, value=4), event("OP", 0, operator=0),
             event("OP", 0, operator=4), event("QUERY", 0), event("QUERY", 1), event("SET", 0, value=3),
             event("QUERY", 0), event("REVOKE", 1), event("OP", 1, operator=3), event("QUERY", 1)])
        self.assertEqual([v for v in solve(ep) if v != IGNORE], [2, 4, 3, UNDEFINED])
        self.assertNotEqual(solve(ep), predict_reversed_order(ep))

    def test_inst_no_implicit_revival(self):
        ep = episode("inst", [event("OP", 0, operator=0), event("QUERY", 0), event("SET", 0, value=1),
                              event("QUERY", 0)])
        self.assertEqual(solve(ep), [IGNORE, UNDEFINED, IGNORE, 1])

    def test_group_identity_empty_prefix(self):
        self.assertEqual(solve(episode("group", [event("QUERY")])), [0])
        self.assertEqual(solve(episode("group", [event("QUERY")], "abelian")), [0])

    def test_all_s5_permutations_and_operators_independent(self):
        for rank, perm in enumerate(itertools.permutations(range(5))):
            for operator, (a, b) in enumerate(itertools.combinations(range(5), 2)):
                ep = episode("group", [event("SET", value=rank), event("OP", operator=operator), event("QUERY")])
                expected = tuple(b if p == a else a if p == b else p for p in perm)
                # Independent expected rank through lexicographic enumeration.
                expected_rank = list(itertools.permutations(range(5))).index(expected)
                self.assertEqual(solve(ep)[-1], expected_rank)

    def test_abelian_inverse_and_order_control(self):
        for task in ("group", "inst"):
            ep = episode(task, [event("SET", value=123), event("OP", operator=0), event("OP", operator=3), event("QUERY")], "abelian")
            self.assertEqual(solve(ep)[-1], 123)
            self.assertEqual(solve(ep), predict_reversed_order(ep))

    def test_s5_order_negative_control_is_detected(self):
        ep = episode("group", [event("SET", value=0), event("OP", operator=0), event("OP", operator=4), event("QUERY")])
        self.assertNotEqual(solve(ep), predict_reversed_order(ep))

    def test_online_reference_agrees_for_all_seeded_tasks(self):
        for task, group in (("mqar", "none"), ("group", "s5"), ("group", "abelian"), ("inst", "s5"), ("inst", "abelian")):
            for seed in SEEDS:
                for split in ("train", "validation", "test"):
                    for ep in generate(task, seed, split, 8, group=group):
                        self.assertEqual(ep["targets"], solve(ep))

    def test_independent_modules_no_generator_import(self):
        text = Path(__file__).with_name("oracle.py").read_text("utf8")
        self.assertNotIn("from .generation import", text)
        self.assertNotIn("import generation", text)

    def test_missing_entity_is_defined_as_undefined(self):
        ep = episode("mqar", [event("WRITE", 0, value=1), event("QUERY", 63)])
        self.assertEqual(solve(ep)[-1], UNDEFINED)


class LeakageTests(unittest.TestCase):
    def test_offline_audit_rejects_stale_serialized_control(self):
        from .audit import audit_episode
        ep = generate("mqar", 11, "train", 1)[0]
        ep["control"][0]["write_id"] = 63
        with self.assertRaises(ValueError):
            audit_episode(ep)

    def test_live_overwrite_and_reinitialization_reported_separately(self):
        from .audit import audit_episode
        ep = episode("mqar", [event("WRITE", 0, value=1), event("WRITE", 0, value=2),
                               event("REVOKE", 0), event("WRITE", 0, value=3), event("QUERY", 0)])
        ep["targets"] = solve(ep)
        audit = audit_episode(ep)
        self.assertEqual(audit["live_overwrite"], 1)
        self.assertEqual(audit["reinitialization_after_revoke"], 1)

    def test_targets_and_provenance_never_enter_view(self):
        ep = generate("inst", 11, "train", 1)[0]
        expected = model_view(ep)
        altered = copy.deepcopy(ep)
        altered["targets"] = [777] * len(ep["events"])
        altered["metadata"] = {"future_answers": "poison", "split": "test"}
        altered["solver"] = "forbidden object"
        self.assertEqual(model_view(altered), expected)
        self.assertEqual(encode_numeric(altered), encode_numeric(ep))
        self.assertEqual(solve(altered), solve(ep))

    def test_future_suffix_cannot_change_prefix_targets_or_features(self):
        for task in ("mqar", "group", "inst"):
            ep = generate(task, 37, "test", 1)[0]
            labels, features = solve(ep), encode_numeric(ep)
            for t, is_query in enumerate(ep["query_mask"]):
                if not is_query:
                    continue
                changed = copy.deepcopy(ep)
                changed["events"][t + 1:] = [event("NOOP") for _ in ep["events"][t + 1:]]
                self.assertEqual(solve(changed)[:t + 1], labels[:t + 1])
                self.assertEqual(encode_numeric(changed)[:t + 1], features[:t + 1])

    def test_future_write_does_not_answer_prior_query(self):
        ep = episode("mqar", [event("QUERY", 2), event("WRITE", 2, value=42), event("QUERY", 2)])
        self.assertEqual(solve(ep), [UNDEFINED, IGNORE, 42])

    def test_event_target_injection_rejected(self):
        ep = generate("mqar", 11, "train", 1)[0]
        ep["events"][0]["target"] = 42
        with self.assertRaises(ValueError):
            model_view(ep)

    def test_control_label_injection_rejected(self):
        view = model_view(generate("mqar", 11, "train", 1)[0])
        view["control"][0]["answer"] = 42
        with self.assertRaises(ValueError):
            validate_model_view(view)

    def test_top_level_label_injection_rejected_at_model_boundary(self):
        view = model_view(generate("mqar", 11, "train", 1)[0])
        view["targets"] = [42] * len(view["events"])
        with self.assertRaises(ValueError):
            validate_model_view(view)

    def test_control_only_current_event_and_no_slot_address_assumption(self):
        ep = generate("inst", 23, "train", 1)[0]
        view = validate_model_view(model_view(ep))
        for ev, control in zip(view["events"], view["control"]):
            self.assertEqual(set(control), CONTROL_FIELDS)
            self.assertEqual(control["read"], ev["kind"] == "QUERY")
            self.assertEqual(control["revoke_id"], ev["entity"] if ev["kind"] == "REVOKE" else -1)
            self.assertNotIn("value", control)
            self.assertNotIn("address", control)

    def test_input_projection_does_not_alias_mutable_events(self):
        ep = generate("mqar", 11, "train", 1)[0]
        view = model_view(ep)
        view["events"][0]["entity"] = 63
        self.assertNotEqual(view["events"], ep["events"])

    def test_neutral_distractors_do_not_act_as_writes(self):
        ep = episode("mqar", [event("WRITE", 0, value=3), event("DISTRACTOR", 0, value=99), event("QUERY", 0)])
        self.assertEqual(solve(ep)[-1], 3)

    def test_entity_renaming_preserves_answers(self):
        for task in ("mqar", "inst"):
            ep = generate(task, 11, "test", 1)[0]
            renamed = copy.deepcopy(ep)
            for ev in renamed["events"]:
                ev["entity"] = 63 - ev["entity"]
            self.assertEqual(solve(ep), solve(renamed))


class GeneratorTests(unittest.TestCase):
    def test_seeded_reproducibility(self):
        for task in ("mqar", "group", "inst"):
            self.assertEqual(generate(task, 11, "train", 4), generate(task, 11, "train", 4))

    def test_split_seeds_and_inputs_disjoint(self):
        derived, inputs = set(), set()
        for seed in SEEDS:
            for split in ("train", "validation", "test"):
                for ep in generate("inst", seed, split, 16):
                    ds, fp = ep["metadata"]["derived_seed"], fingerprint(model_view(ep))
                    self.assertNotIn(ds, derived); derived.add(ds)
                    self.assertNotIn(fp, inputs); inputs.add(fp)

    def test_prefix_samples_count_stable(self):
        self.assertEqual(generate("mqar", 11, "train", 1), generate("mqar", 11, "train", 4)[:1])

    def test_exact_event_and_query_counts(self):
        for task in ("mqar", "group", "inst"):
            ep = generate(task, 11, "test", 1)[0]
            cfg = ep["metadata"]["difficulty"]
            counts = Counter(e["kind"] for e in ep["events"])
            self.assertEqual(len(ep["events"]), cfg["length"])
            self.assertEqual(counts["DISTRACTOR"], cfg["distractors"])
            self.assertEqual(counts["REVOKE"], cfg["revocations"])
            self.assertEqual(counts["WRITE" if task == "mqar" else "SET"], cfg["entities"] + cfg["overwrite"])
            self.assertEqual(counts["QUERY"], cfg["operations"] if task == "mqar" else cfg["queries"])
            self.assertEqual(counts["OP"], 0 if task == "mqar" else cfg["operations"])
            self.assertEqual(ep["query_mask"], [t != IGNORE for t in ep["targets"]])

    def test_length_axis_preserves_action_stream(self):
        for task in ("mqar", "group", "inst"):
            a = generate(task, 11, "test", 1)[0]
            b = generate(task, 11, "test", 1, length=40)[0]
            meaningful = lambda e: [ev for ev in e["events"] if ev["kind"] not in ("NOOP", "DISTRACTOR")]
            self.assertEqual(meaningful(a), meaningful(b))
            self.assertEqual([v for v in a["targets"] if v != IGNORE], [v for v in b["targets"] if v != IGNORE])

    def test_budget_constraints_do_not_silently_relax(self):
        for kwargs in (dict(length=2), dict(entities=0), dict(entities=65), dict(operations=-1), dict(length=513)):
            with self.assertRaises(ValueError):
                generate("mqar", 11, "train", 1, **kwargs)

    def test_unknown_axis_and_split_fail(self):
        for args in (("mqar", 11, "test", 1, {"L": 24}), ("mqar", 11, "val", 1, {}),
                     ("unknown", 11, "test", 1, {}), ("mqar", 11, "train", -1, {})):
            with self.assertRaises(ValueError):
                generate(*args[:4], **args[4])

    def test_ordered_pair_partitions_disjoint_and_exhaustive(self):
        sets = [pair_partition(s) for s in ("train", "validation", "test")]
        self.assertEqual([len(s) for s in sets], [60, 20, 20])
        for a, b in itertools.combinations(sets, 2):
            self.assertFalse(a & b)
        self.assertEqual(len(set.union(*(set(s) for s in sets))), 100)

    def test_generated_ordered_pairs_obey_partitions(self):
        found = 0
        for task in ("group", "inst"):
            for split in ("train", "validation", "test"):
                allowed = pair_partition(split)
                for ep in generate(task, 11, split, 32, operations=8, regime="ordered_pairs"):
                    previous = {}
                    for ev in ep["events"]:
                        ent = ev["entity"]
                        if ev["kind"] in ("SET", "REVOKE"):
                            previous.pop(ent, None)
                        if ev["kind"] == "OP":
                            if ent in previous:
                                self.assertIn((previous[ent], ev["operator"]), allowed)
                                found += 1
                            previous[ent] = ev["operator"]
                    self.assertEqual(ep["targets"], solve(ep))
        self.assertGreater(found, 100)

    def test_cross_group_is_separate(self):
        for split in ("train", "validation", "test"):
            ep = generate("group", 11, split, 1, regime="cross_group")[0]
            self.assertEqual(ep["group"], "abelian" if split == "test" else "s5")
            self.assertEqual(solve(ep), ep["targets"])
        with self.assertRaises(ValueError):
            generate("mqar", 11, "test", 1, regime="cross_group")

    def test_within_group_profiles_change_one_axis_only(self):
        table = profiles()
        for row in table:
            base = next(r["difficulty"] for r in table if r["task"] == row["task"] and r["profile"] == "base")
            if row["evaluation"] == "within_group_one_axis":
                changed = [k for k in base if base[k] != row["difficulty"][k]]
                self.assertEqual(changed, [row["changed_axis"]])
            for split in ("train", "validation", "test"):
                ep = generate(row["task"], 11, split, 1, **row["difficulty"])[0]
                self.assertEqual(ep["targets"], solve(ep))

    def test_compact_six_class_inst_targets(self):
        ep = generate("inst", 11, "test", 1)[0]
        self.assertEqual(label_vocabulary("inst", "s5"), [0, 1, 2, 3, 4, 125])
        labels = compact_targets(ep)
        self.assertTrue(all(t == IGNORE or 0 <= t < 6 for t in labels))
        for raw, compact in zip(ep["targets"], labels):
            self.assertEqual(compact, 5 if raw == UNDEFINED else raw)

    def test_numeric_encoding_and_thread_cap(self):
        ep = generate("inst", 11, "test", 1)[0]
        self.assertEqual(len(encode_numeric(ep)), 24)
        self.assertTrue(all(len(row) == 8 for row in encode_numeric(ep)))
        with self.assertRaises(ValueError):
            encode_numeric(ep, dim=4)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.assertLessEqual(int(os.environ[name]), 2)

    def test_empty_batch_and_invalid_state_domains(self):
        self.assertEqual(generate("mqar", 11, "train", 0), [])
        for ep in (episode("inst", [event("SET", value=6)]), episode("group", [event("SET", value=120)])):
            with self.assertRaises(ValueError):
                solve(ep)


class ProtocolTests(unittest.TestCase):
    def cells(self):
        return [{"arm": arm, "seed": seed, "task": "inst", "profile": "base", "status": "complete",
                 "accuracy": 0.4 + (0.01 * i if arm == "M0" else 0.10 + 0.02 * i)}
                for arm in ARMS for i, seed in enumerate(SEEDS)]

    def test_five_paired_seed_t_interval(self):
        result = paired_summary(self.cells())
        self.assertTrue(result["inference_defined"])
        self.assertEqual(result["df"], 4)
        self.assertAlmostEqual(result["mean_delta"], 0.12)
        self.assertLess(result["ci95"][0], result["mean_delta"])
        self.assertGreater(result["ci95"][1], result["mean_delta"])

    def test_failures_censored_unattempted_never_dropped(self):
        cells = self.cells()
        for c in cells:
            if c["arm"] == "M3" and c["seed"] in (11, 23, 37):
                c["status"] = {11: "failed", 23: "censored", 37: "unattempted"}[c["seed"]]
                c["accuracy"] = 0.99 if c["seed"] == 23 else None
                c["reason"] = "deliberate negative test"
        result = paired_summary(cells)
        self.assertEqual(len(result["seeds"]), 5)
        self.assertEqual(result["complete_pairs"], 2)
        self.assertIsNone(result["mean_delta"])
        self.assertIsNone(result["ci95"])
        self.assertLess(result["all_seed_mean_bounds"][0], result["all_seed_mean_bounds"][1])

    def test_absent_seed_becomes_unattempted_not_replacement(self):
        cells = [c for c in self.cells() if not (c["arm"] == "M3" and c["seed"] == 11)]
        result = paired_summary(cells)
        self.assertFalse(result["inference_defined"])
        self.assertEqual(result["seeds"][0]["candidate"]["status"], "unattempted")

    def test_all_unattempted_bounds(self):
        result = paired_summary([])
        self.assertEqual(result["all_seed_mean_bounds"], [-1, 1])
        self.assertEqual(result["complete_pairs"], 0)

    def test_duplicates_and_pooling_rejected(self):
        cells = self.cells()
        with self.assertRaises(ValueError):
            paired_summary(cells + [cells[0]])
        changed = copy.deepcopy(cells)
        changed[0]["task"] = "mqar"
        with self.assertRaises(ValueError):
            paired_summary(changed)

    def test_gap_undefined_without_epsilon_division(self):
        for baseline, ceiling in ((1, 1), (0.999, 1), (0.5, 0.49), (0.99, 0.999), (0.99, 1)):
            result = gap_recovery(0.999, baseline, ceiling)
            self.assertFalse(result["defined"])
            self.assertIsNone(result["value"])
        self.assertAlmostEqual(gap_recovery(0.75, 0.5)["value"], 0.5)

    def test_absolute_accuracy_requires_all_five_seeds(self):
        result = paired_summary(self.cells())
        self.assertAlmostEqual(result["absolute_accuracy"]["baseline"]["mean_accuracy"], 0.42)
        cells = [c for c in self.cells() if not (c["arm"] == "M3" and c["seed"] == 11)]
        result = paired_summary(cells)
        self.assertIsNone(result["absolute_accuracy"]["candidate"]["mean_accuracy"])

    def test_gap_invalid_values_rejected(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            with self.assertRaises(ValueError):
                gap_recovery(value, 0.5)

    def test_accuracy_includes_undefined_and_never_drops_failures(self):
        result = accuracy([IGNORE, 1, 3, 7], [IGNORE, 1, UNDEFINED, 7])
        self.assertEqual(result["queries"], 3)
        self.assertEqual(result["accuracy"], 2 / 3)
        self.assertIsNone(accuracy([IGNORE], [IGNORE])["accuracy"])

    def test_budget_enforces_threads_updates_time_and_total(self):
        cells = self.cells()
        for c in cells:
            c.update(wall_seconds=31, training_steps=20, cpu_threads=2, hyperparameter_settings=1,
                     parameter_bytes=128, persistent_state_bytes=128, cache_bytes=0,
                     reserved_workspace_bytes=256, retrieval_operations=100, latency_seconds=0.1)
        report = check_budget(cells)
        self.assertFalse(report["within_known_limits"])
        self.assertTrue(report["fully_measured"])
        cells[0].update(wall_seconds=91, training_steps=61, cpu_threads=3, hyperparameter_settings=2)
        self.assertGreaterEqual(len(check_budget(cells)["violations"]), 5)

    def test_unmeasured_budget_not_claimed_matched(self):
        report = check_budget(self.cells())
        self.assertFalse(report["fully_measured"])
        self.assertTrue(report["unknown"])

    def test_factorial_missing_cells_preserved(self):
        cells = self.cells()
        result = factorial_summary(cells)
        self.assertTrue(result["interaction_defined"])
        cells = [c for c in cells if not (c["arm"] == "M2" and c["seed"] == 71)]
        result = factorial_summary(cells)
        self.assertFalse(result["interaction_defined"])
        self.assertIsNone(result["interaction_mean"])
        self.assertEqual(result["interaction_seeds"][-1]["statuses"]["M2"], "unattempted")

    def test_undefined_constant_control_cannot_pass_defined_fixture(self):
        ep = episode("mqar", [event("WRITE", 0, value=7), event("QUERY", 0)])
        self.assertEqual(accuracy(predict_undefined(ep), solve(ep))["accuracy"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
