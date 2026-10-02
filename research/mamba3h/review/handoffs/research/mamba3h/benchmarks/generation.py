"""Seeded generators with online causal labels; no imports from oracle.py."""

import hashlib
import itertools
import random

from .schema import IGNORE, UNDEFINED, VERSION, event, fingerprint, model_view

SEEDS = (11, 23, 37, 53, 71)
SPLITS = ("train", "validation", "test")
TRANSPOSES = tuple(itertools.combinations(range(5), 2))
PERMS = tuple(itertools.permutations(range(5)))
PERM_IDS = {p: i for i, p in enumerate(PERMS)}
ABELIAN_OPS = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (4, 0, 0), (0, 4, 0),
               (0, 0, 4), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1))
DEFAULTS = dict(length=24, entities=4, operations=4, distractors=4,
                overwrite=2, revocations=1, queries=3, group="s5", regime="within_group")


def pair_partition(split):
    """Disjoint ordered-edge partitions with a nonempty successor for every op."""
    offsets = {"train": range(6), "validation": (6, 7), "test": (8, 9)}[split]
    return frozenset((a, (a + d) % 10) for a in range(10) for d in offsets)


def derived_seed(task, seed, split, sample, regime):
    text = f"{VERSION}|{task}|{seed}|{split}|{sample}|{regime}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:16], "big")


def _digits(value):
    return (value // 25, (value // 5) % 5, value % 5)


def _encode(state, task, group):
    if state is None:
        return UNDEFINED
    if group == "s5":
        return PERM_IDS[state] if task == "group" else state
    return state[0] * 25 + state[1] * 5 + state[2]


def _act(state, op, task, group):
    if state is None:
        return None
    if group == "abelian":
        return tuple((a + b) % 5 for a, b in zip(state, ABELIAN_OPS[op]))
    a, b = TRANSPOSES[op]
    def swap(v):
        return b if v == a else a if v == b else v
    return tuple(swap(v) for v in state) if task == "group" else swap(state)


def _online_labels(task, group, events):
    states = {0: tuple(range(5)) if group == "s5" else (0, 0, 0)} if task == "group" else {}
    labels = []
    for ev in events:
        kind, entity = ev["kind"], ev["entity"]
        if kind in ("WRITE", "SET"):
            raw = ev["value"]
            states[entity] = raw if task == "mqar" or (task == "inst" and group == "s5") else (
                PERMS[raw] if group == "s5" else _digits(raw))
        elif kind == "OP":
            states[entity] = _act(states.get(entity), ev["operator"], task, group)
        elif kind == "REVOKE":
            states[entity] = None
        if kind == "QUERY":
            value = states.get(entity)
            labels.append(UNDEFINED if value is None else value if task == "mqar" else _encode(value, task, group))
        else:
            labels.append(IGNORE)
    return labels


def _validate(task, seed, split, count, cfg):
    if task not in ("mqar", "group", "inst") or split not in SPLITS:
        raise ValueError("Unknown task or split")
    if type(seed) is not int or type(count) is not int or not 0 <= count <= 4096:
        raise ValueError("Integer seed and count in [0,4096] required")
    for name in ("length", "entities", "operations", "distractors", "overwrite", "revocations", "queries"):
        if type(cfg[name]) is not int or cfg[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not 1 <= cfg["entities"] <= 64 or not 1 <= cfg["length"] <= 512:
        raise ValueError("entities [1,64] and length [1,512] required")
    if task == "group" and cfg["entities"] != 1:
        raise ValueError("group task requires entities=1; use INST for multi-entity tracking")
    if cfg["group"] not in ("none", "s5", "abelian"):
        raise ValueError("Unknown group")
    if (task == "mqar") != (cfg["group"] == "none"):
        raise ValueError("MQAR uses group=none; group/INST use s5 or abelian")
    if cfg["regime"] not in ("within_group", "ordered_pairs", "cross_group"):
        raise ValueError("Unknown regime")
    if task == "mqar" and cfg["regime"] != "within_group":
        raise ValueError("Ordered/cross-group regimes are for group and INST")
    required = cfg["entities"] + cfg["operations"] + cfg["distractors"] + cfg["overwrite"] + cfg["revocations"]
    if task != "mqar":
        required += cfg["queries"]
    if required > cfg["length"]:
        raise ValueError(f"Event budget {required} exceeds length {cfg['length']}; axes are never silently changed")


def generate(task, seed, split, count, **difficulty):
    """Return examples with events/tokens, separate targets, mask and control.

    operations: MQAR READ count; group/INST OP count. overwrite: extra WRITE/SET
    count, revocations: REVOKE count, distractors: nonmutating DISTRACTOR count.
    NOOP fills the remaining length. All counters are exact and independent.
    Entity count is a universe size, not a persistent memory capacity.
    """
    unknown = set(difficulty) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown axes: {sorted(unknown)}")
    cfg = dict(DEFAULTS, **difficulty)
    if task == "mqar":
        cfg["group"] = difficulty.get("group", "none")
    if task == "group":
        cfg["entities"] = difficulty.get("entities", 1)
    _validate(task, seed, split, count, cfg)
    examples = []
    for sample in range(count):
        effective_group = cfg["group"]
        if cfg["regime"] == "cross_group":
            effective_group = "abelian" if split == "test" else "s5"
        sub_seed = derived_seed(task, seed, split, sample, cfg["regime"])
        rng = random.Random(sub_seed)
        value_limit = 5 if task == "inst" and effective_group == "s5" else 120 if effective_group == "s5" else 125
        initial_kind = "WRITE" if task == "mqar" else "SET"
        events = [event(initial_kind, entity, value=rng.randrange(value_limit)) for entity in range(cfg["entities"])]
        schedules = (["QUERY"] * cfg["operations"] if task == "mqar" else
                     ["OP"] * cfg["operations"] + ["QUERY"] * cfg["queries"])
        schedules += [initial_kind] * cfg["overwrite"] + ["REVOKE"] * cfg["revocations"]
        rng.shuffle(schedules)
        previous = {}
        pairs = pair_partition(split) if cfg["regime"] == "ordered_pairs" else None
        for kind in schedules:
            entity = rng.randrange(cfg["entities"])
            if kind == "OP":
                choices = range(10) if entity not in previous or pairs is None else [b for b in range(10) if (previous[entity], b) in pairs]
                op = rng.choice(list(choices))
                previous[entity] = op
                events.append(event(kind, entity, operator=op))
            else:
                events.append(event(kind, entity, value=rng.randrange(value_limit) if kind == initial_kind else None))
                if kind in (initial_kind, "REVOKE"):
                    previous.pop(entity, None)
        # Neutral-event placement uses a distinct RNG; increasing L does not change
        # the chronological non-neutral stream or its answers.
        pad_rng = random.Random(sub_seed ^ 0x5A5A5A)
        neutral = [event("DISTRACTOR", pad_rng.randrange(cfg["entities"]), value=pad_rng.randrange(125))
                   for _ in range(cfg["distractors"])]
        neutral += [event("NOOP") for _ in range(cfg["length"] - len(events) - len(neutral))]
        # Insert at slots while preserving the full causal action order.
        slots = sorted(pad_rng.sample(range(cfg["length"]), len(events)))
        event_it, neutral_it, slot_set = iter(events), iter(neutral), set(slots)
        events = [next(event_it) if t in slot_set else next(neutral_it) for t in range(cfg["length"])]
        labels = _online_labels(task, effective_group, events)
        episode = {"schema": VERSION, "task": task, "group": effective_group, "events": events}
        view = model_view(episode)
        episode.update(tokens=view["tokens"], control=view["control"], query_mask=view["query_mask"], targets=labels)
        episode["metadata"] = {"seed": seed, "split": split, "sample": sample, "derived_seed": str(sub_seed),
                               "difficulty": cfg, "effective_group": effective_group,
                               "model_input_sha256": fingerprint(view)}
        examples.append(episode)
    return examples
