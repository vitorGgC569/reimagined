"""Label-free model input and current-event routing metadata.

Undefined is an explicit supervised class, never an omitted observation.
"""

import hashlib
import json
import math

VERSION = "mamba3h.benchmarks.v1"
UNDEFINED = 125
IGNORE = -100
KINDS = ("WRITE", "SET", "OP", "QUERY", "REVOKE", "NOOP", "DISTRACTOR")
TASKS = ("mqar", "group", "inst")
GROUPS = ("none", "s5", "abelian")
EVENT_FIELDS = {"kind", "entity", "operator", "value"}
CONTROL_FIELDS = {"kind", "entity", "write_id", "revoke_id", "write", "read"}


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")


def fingerprint(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def event(kind, entity=0, operator=None, value=None):
    return {"kind": kind, "entity": entity, "operator": operator, "value": value}


def validate_events(events):
    for ev in events:
        if set(ev) != EVENT_FIELDS or ev["kind"] not in KINDS:
            raise ValueError("Invalid event fields; labels/provenance must not enter events")
        if type(ev["entity"]) is not int or not 0 <= ev["entity"] < 64:
            raise ValueError("entity must be an integer in [0,64)")
        expected = (ev["kind"] == "OP", ev["kind"] in ("WRITE", "SET", "DISTRACTOR"))
        for name, present, limit in (("operator", expected[0], 10), ("value", expected[1], 125)):
            item = ev[name]
            if present and (type(item) is not int or not 0 <= item < limit):
                raise ValueError(f"{name} must be an integer in [0,{limit})")
            if not present and item is not None:
                raise ValueError(f"Unexpected {name} on {ev['kind']}")


def causal_control(ev):
    """Only routing/identity metadata; observed values stay in token features."""
    return {
        "kind": ev["kind"], "entity": ev["entity"],
        "write_id": ev["entity"] if ev["kind"] in ("WRITE", "SET", "OP") else -1,
        "revoke_id": ev["entity"] if ev["kind"] == "REVOKE" else -1,
        "write": ev["kind"] in ("WRITE", "SET", "OP"), "read": ev["kind"] == "QUERY",
    }


def model_view(episode):
    """Whitelist projection; never forward targets, seed, split or solver objects."""
    task, group = episode["task"], episode["group"]
    if task not in TASKS or group not in GROUPS:
        raise ValueError("Unknown task/group")
    events = [dict(ev) for ev in episode["events"]]
    validate_events(events)
    if task == "mqar" and (group != "none" or any(e["kind"] in ("SET", "OP") for e in events)):
        raise ValueError("MQAR only accepts none/WRITE/QUERY/REVOKE/neutral events")
    if task != "mqar" and (group == "none" or any(e["kind"] == "WRITE" for e in events)):
        raise ValueError("Group/INST require SET/OP and a concrete group")
    if task == "group" and any(e["entity"] != 0 for e in events):
        raise ValueError("Group tracking has one stream, entity zero")
    if group == "s5" and any(e["kind"] == "SET" and e["value"] >= (120 if task == "group" else 5) for e in events):
        raise ValueError("S5 SET value outside the task state space")
    tokens = [[KINDS.index(e["kind"]), e["entity"],
               0 if e["operator"] is None else e["operator"] + 1,
               0 if e["value"] is None else e["value"] + 1] for e in events]
    return {"schema": VERSION, "task": task, "group": group, "events": events,
            "tokens": tokens, "control": [causal_control(e) for e in events],
            "query_mask": [e["kind"] == "QUERY" for e in events]}


def validate_model_view(view):
    expected = {"schema", "task", "group", "events", "tokens", "control", "query_mask"}
    if set(view) != expected or view != model_view(view):
        raise ValueError("Model view is not the exact label-free causal projection")
    return view


def encode_numeric(episode, dim=8):
    """Deterministic event-local floats [L,D]; callers form CPU x[B,D].

    This is a fixed input encoding, not a learned embedding or a Mamba baseline.
    No query answer, future event, split id or RNG seed is read.
    """
    if type(dim) is not int or dim < 8:
        raise ValueError("dim must be >=8")
    view = model_view(episode)
    result = []
    for kind, entity, op, value in view["tokens"]:
        row = [kind / 6, entity / 63, op / 10, value / 125,
               TASKS.index(view["task"]) / 2, GROUPS.index(view["group"]) / 2,
               float(kind == KINDS.index("QUERY")), float(kind == KINDS.index("REVOKE"))]
        row.extend(math.sin((j + 1) * (kind + 1) + entity + op + value) for j in range(dim - 8))
        result.append(row)
    return result
