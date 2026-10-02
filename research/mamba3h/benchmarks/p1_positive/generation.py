"""P1fresh namespaced sampling; global observable-input rejection is bounded."""

import hashlib
import random

from .schema import COUNTS, DIFFICULTY, IGNORE, NAMESPACE, SEEDS, VERSION, event, fingerprint, model_numeric, model_view, validate_profile

MAX_ATTEMPTS = 16


class DuplicateBudgetExceeded(RuntimeError):
    pass


def data_seed(seed, split, sample, attempt=0):
    key = f"{NAMESPACE}|{seed}|{split}|{sample}|attempt={attempt}"
    return int.from_bytes(hashlib.sha256(key.encode("ascii")).digest()[:16], "big")


def input_fingerprint(ep):
    # Hash the actual x/routing seen by models, not provenance or target fields.
    view = model_view(ep)
    return fingerprint({"x": model_numeric(ep), "control": view["control"], "query_mask": view["query_mask"]})


def _candidate(seed, split, sample, attempt):
    derived = data_seed(seed, split, sample, attempt)
    rng = random.Random(derived)
    entities = rng.sample(range(4), 3)
    prefix = [event("WRITE", entity, rng.randrange(4)) for entity in entities]
    prefix += [event("NOOP", rng.randrange(4)), event("NOOP", rng.randrange(4))]
    rng.shuffle(prefix)
    queried = rng.choice(entities)
    events = prefix + [event("QUERY", queried)]
    # Online chronological labeler, independent of oracle's backward scan.
    bindings = {}
    targets = []
    for e in events:
        if e["kind"] == "WRITE":
            bindings[e["entity"]] = e["value"]
        targets.append(bindings[e["entity"]] if e["kind"] == "QUERY" else IGNORE)
    ep = {"schema": VERSION, "task": "mqar", "group": "none", "events": events, "targets": targets}
    view = model_view(ep)
    ep.update(tokens=view["tokens"], control=view["control"], query_mask=view["query_mask"])
    ep["metadata"] = {"namespace": NAMESPACE, "model_seed": seed, "seed": seed, "split": split,
        "sample": sample, "attempt": attempt, "data_seed": str(derived), "difficulty": dict(DIFFICULTY),
        "input_sha256": input_fingerprint(ep), "event_input_sha256": fingerprint(view)}
    validate_profile(ep)
    return ep


def _accept(candidate_factory, seen, max_attempts=MAX_ATTEMPTS):
    """The registry is shared across all seeds/splits; failure cannot relax it."""
    for attempt in range(max_attempts):
        ep = candidate_factory(attempt)
        digest = input_fingerprint(ep)
        if digest in seen:
            continue
        seen.add(digest)
        return ep, attempt
    raise DuplicateBudgetExceeded(f"No unique observable input after {max_attempts} attempts")


def build_corpus():
    """Canonical 800-example corpus; every API call uses the same global registry.

    Order: fixed seeds, train/validation/test, then sample index. No cache is
    stored in model state or shared across train/test model calls.
    """
    corpus, seen, attempts = {}, set(), 0
    rejected = []
    for seed in SEEDS:
        for split, count in COUNTS.items():
            examples = []
            for sample in range(count):
                ep, attempt = _accept(lambda a: _candidate(seed, split, sample, a), seen)
                attempts += attempt + 1
                if attempt:
                    rejected.append({"seed": seed, "split": split, "sample": sample, "rejected_attempts": attempt})
                examples.append(ep)
            corpus[(seed, split)] = examples
    return corpus, {"accepted": len(seen), "attempts": attempts, "rejected": rejected,
                    "max_attempts_per_example": MAX_ATTEMPTS, "global_rejection": True}


def generate(task, seed, split, count, **difficulty):
    """Small fixed fresh corpus API, independent of call order/count.

    Rebuild the canonical bounded registry even for a requested subset, ensuring
    separate generate calls cannot reintroduce an inter-split duplicate.
    """
    if task != "mqar" or type(seed) is not int or seed not in SEEDS or split not in COUNTS:
        raise ValueError("P1fresh supports MQAR, five registered model seeds, and train/validation/test")
    if type(count) is not int or not 0 <= count <= COUNTS[split]:
        raise ValueError(f"count must be 0..{COUNTS[split]}")
    if any(key not in DIFFICULTY or value != DIFFICULTY[key] for key, value in difficulty.items()):
        raise ValueError("P1 positive difficulty is frozen; alternate axes need a new protocol")
    if count == 0:
        return []
    corpus, _ = build_corpus()
    return corpus[(seed, split)][:count]
