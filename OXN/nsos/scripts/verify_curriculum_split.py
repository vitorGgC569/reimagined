#!/usr/bin/env python3
"""Regression guard for the curriculum train/eval split.

build_curriculum() calls the phase builders with DIFFERENT seeds for the
train and eval passes (seed+phase*1000+11 vs +29).  The split helpers must
therefore assign every source item to train/eval by CONTENT, not by a
per-call RNG shuffle — otherwise train[:cut] and eval[cut:] are slices of
different permutations and the same item leaks into both sides.

This script exercises the exact failure scenario (two different RNGs) and
asserts the train and eval selections are disjoint and jointly cover the
corpus.  It is pure-stdlib and fast; run it as a gate before trusting any
held-out / perplexity number from the curriculum harness.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nsos_curriculum_lib as lib  # noqa: E402


def _check_rows() -> None:
    # Mix of rows WITH an explicit id and rows WITHOUT one (canonical-json key).
    rows = [{"id": f"row-{i}", "text": f"content {i}"} for i in range(500)]
    rows += [{"text": f"anon {i}", "meta": {"k": i}} for i in range(120)]

    # The two calls use DIFFERENT seeds, exactly like build_curriculum.
    train = lib._pick_split_rows(rows, "train", random.Random(1337 + 11))
    eval_ = lib._pick_split_rows(rows, "eval", random.Random(1337 + 29))

    def key(r):
        return lib._row_split_key(r)

    train_keys = {key(r) for r in train}
    eval_keys = {key(r) for r in eval_}
    all_keys = {key(r) for r in rows}

    overlap = train_keys & eval_keys
    assert not overlap, f"LEAK: {len(overlap)} rows in BOTH train and eval"
    assert train_keys | eval_keys == all_keys, "rows split does not cover corpus"
    assert train_keys and eval_keys, "one side is empty"
    frac = len(train_keys) / len(all_keys)
    assert 0.70 <= frac <= 0.92, f"train fraction off: {frac:.3f}"
    print(f"rows: OK  train={len(train_keys)} eval={len(eval_keys)} frac={frac:.3f} overlap=0")


def _check_subset() -> None:
    items = [(f"title-{i}", f"chunk body number {i}", "wikipedia") for i in range(800)]

    train = lib._pick_split_subset(items, "train", random.Random(2024 + 11))
    eval_ = lib._pick_split_subset(items, "eval", random.Random(2024 + 29))

    train_keys = {"\x1f".join(map(str, t)) for t in train}
    eval_keys = {"\x1f".join(map(str, t)) for t in eval_}
    all_keys = {"\x1f".join(map(str, t)) for t in items}

    overlap = train_keys & eval_keys
    assert not overlap, f"LEAK: {len(overlap)} chunks in BOTH train and eval"
    assert train_keys | eval_keys == all_keys, "subset split does not cover corpus"
    assert train_keys and eval_keys, "one side is empty"
    frac = len(train_keys) / len(all_keys)
    assert 0.70 <= frac <= 0.90, f"train fraction off: {frac:.3f}"
    print(f"subset: OK  train={len(train_keys)} eval={len(eval_keys)} frac={frac:.3f} overlap=0")


def _check_determinism() -> None:
    # Same content, different seed -> SAME assignment (content-addressed).
    rows = [{"id": f"r{i}"} for i in range(300)]
    a = {lib._row_split_key(r) for r in lib._pick_split_rows(rows, "train", random.Random(1))}
    b = {lib._row_split_key(r) for r in lib._pick_split_rows(rows, "train", random.Random(999999))}
    assert a == b, "train assignment changed with the seed (must be content-addressed)"
    print("determinism: OK  train assignment is seed-independent")


def main() -> int:
    _check_rows()
    _check_subset()
    _check_determinism()
    print("ALL SPLIT CHECKS PASSED (train/eval are disjoint and seed-independent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
