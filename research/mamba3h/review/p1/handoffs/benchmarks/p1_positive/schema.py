"""Current-event one-hot input, routing whitelist, and fixed P1 profile."""

import hashlib
import json

VERSION = "mamba3h.p1_positive.v1"
NAMESPACE = "P1fresh-MQAR-positive-v1"
IGNORE = -100
KINDS = ("WRITE", "NOOP", "QUERY")
EVENT_FIELDS = {"kind", "entity", "operator", "value"}
CONTROL_FIELDS = {"kind", "entity", "write_id", "revoke_id", "write", "read"}
DIFFICULTY = dict(length=6, entities=4, values=4, writes=3, distractors=2, queries=1)
COUNTS = {"train": 64, "validation": 32, "test": 64}
SEEDS = (11, 23, 37, 53, 71)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")


def fingerprint(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def event(kind, entity, value=None):
    return {"kind": kind, "entity": entity, "operator": None, "value": value}


def validate_events(events):
    for e in events:
        if not isinstance(e, dict) or set(e) != EVENT_FIELDS or e["kind"] not in KINDS or e["operator"] is not None:
            raise ValueError("P1 event fields/kind invalid; no target, future or provenance allowed")
        if type(e["entity"]) is not int or not 0 <= e["entity"] < 4:
            raise ValueError("Entity must be 0..3")
        if e["kind"] == "WRITE":
            if type(e["value"]) is not int or not 0 <= e["value"] < 4:
                raise ValueError("WRITE value must be 0..3")
        elif e["value"] is not None:
            raise ValueError("QUERY/NOOP cannot expose any value or answer")


def causal_control(e):
    return {"kind": e["kind"], "entity": e["entity"],
            "write_id": e["entity"] if e["kind"] == "WRITE" else -1,
            "revoke_id": -1, "write": e["kind"] == "WRITE", "read": e["kind"] == "QUERY"}


def model_view(ep):
    """Project only events/current routing. Labels/provenance are never inputs."""
    if ep.get("task") != "mqar" or ep.get("group", "none") != "none":
        raise ValueError("This P1 profile is only MQAR, group=none")
    events = [dict(e) for e in ep["events"]]
    validate_events(events)
    return {"schema": VERSION, "task": "mqar", "group": "none", "events": events,
            "tokens": [[KINDS.index(e["kind"]), e["entity"], 0 if e["value"] is None else e["value"] + 1] for e in events],
            "control": [causal_control(e) for e in events],
            "query_mask": [e["kind"] == "QUERY" for e in events]}


def validate_model_view(view):
    if set(view) != {"schema", "task", "group", "events", "tokens", "control", "query_mask"} or view != model_view(view):
        raise ValueError("Invalid or poisoned model view")
    return view


def model_numeric(ep):
    """[L,8] floats: one-hot entity4, then WRITE-only observed value4.

    NOOP and QUERY with the same id have identical rows (only current identity).
    Their write/read distinction is in causal routing, not an answer feature.
    """
    result = []
    for e in model_view(ep)["events"]:
        result.append([float(j == e["entity"]) for j in range(4)] +
                      [float(e["kind"] == "WRITE" and j == e["value"]) for j in range(4)])
    return result


def encode_numeric(ep, dim=8):
    if dim != 8:
        raise ValueError("Frozen P1 positive encoding is exactly D=8")
    return model_numeric(ep)


def validate_profile(ep):
    view = model_view(ep)
    events = view["events"]
    if len(events) != 6 or events[-1]["kind"] != "QUERY":
        raise ValueError("Exactly L6 and final QUERY required")
    writes = [e for e in events[:5] if e["kind"] == "WRITE"]
    if len(writes) != 3 or len({e["entity"] for e in writes}) != 3:
        raise ValueError("Three distinct-entity WRITEs required")
    if sum(e["kind"] == "NOOP" for e in events[:5]) != 2:
        raise ValueError("Exactly two entity-id NOOPs required")
    if events[-1]["entity"] not in {e["entity"] for e in writes}:
        raise ValueError("Final QUERY must reference a written entity")
    if view["query_mask"] != [False] * 5 + [True]:
        raise ValueError("Exactly one final current query required")
    return ep
