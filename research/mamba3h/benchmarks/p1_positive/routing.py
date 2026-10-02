"""Exact positive CPU routing reference, not a learned or native model.

Input is only current x[8] and routing control. Keys and values are copied from
preread x. Capacity=3, chronological commit, exact top1, no-read skips work.
The test harness alone decodes output and compares with the independent solver.
"""

import math
from dataclasses import dataclass

from .schema import CONTROL_FIELDS


@dataclass
class State:
    slots: list

    @classmethod
    def empty(cls):
        return cls([None, None, None])


def step(x, state, *, control):
    if len(x) != 8 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in x):
        raise ValueError("Current finite x[8] required")
    if set(control) != CONTROL_FIELDS:
        raise ValueError("No target/value/future or arbitrary control fields allowed")
    kind, entity = control["kind"], control["entity"]
    if kind not in ("WRITE", "NOOP", "QUERY") or type(entity) is not int or not 0 <= entity < 4:
        raise ValueError("Invalid current event")
    if control != {"kind": kind, "entity": entity, "write_id": entity if kind == "WRITE" else -1,
                   "revoke_id": -1, "write": kind == "WRITE", "read": kind == "QUERY"}:
        raise ValueError("Routing gates must match current event metadata")
    key, value = tuple(x[:4]), tuple(x[4:])  # Pre-retrieval projection of observed x.
    if sum(key) != 1 or any(v not in (0, 1) for v in key) or key[entity] != 1:
        raise ValueError("Current key is not the indicated one-hot identity")
    if kind == "WRITE":
        if sum(value) != 1 or any(v not in (0, 1) for v in value):
            raise ValueError("WRITE must carry an observed one-hot value")
    elif any(value):
        raise ValueError("Neutral/query x must not carry a value or target")
    if len(state.slots) != 3:
        raise ValueError("Fixed positive reference capacity is three")
    slots = list(state.slots)
    y = [0.0] * 8
    comparisons, reads, writes = 0, 0, 0
    if control["read"]:
        scored = [(sum(a * b for a, b in zip(key, slot[1])), j)
                  for j, slot in enumerate(slots) if slot is not None]
        comparisons = len(scored)
        reads = 1
        if scored:
            _, selected = max(scored, key=lambda item: (item[0], -item[1]))
            if scored and slots[selected][0] == entity:
                y = list(slots[selected][1] + slots[selected][2])
    if control["write"]:
        matching = [j for j, slot in enumerate(slots) if slot is not None and slot[0] == entity]
        free = [j for j, slot in enumerate(slots) if slot is None]
        if not matching and not free:
            raise ValueError("Capacity exhausted; reference does not silently evict")
        destination = matching[0] if matching else free[0]
        slots[destination] = (entity, key, value)
        writes = 1
    return y, State(slots), {"capacity": 3, "occupancy": sum(s is not None for s in slots),
        "similarity_comparisons": comparisons, "read_operations": reads, "write_operations": writes,
        "parameter_bytes": 0, "logical_reserved_state_bytes": 3 * (8 * 8 + 8), "cache_bytes": 0,
        "byte_measurement": "logical_float64_vectors_plus_int64_identity_not_Python_object_RSS"}


def retrieve(ep):
    """Diagnostic only; reads model_view/current numeric x, never target arrays."""
    from .schema import model_numeric, model_view
    view = model_view(ep)
    state, outputs, stats = State.empty(), [], []
    for x, control in zip(model_numeric(ep), view["control"]):
        y, state, current = step(x, state, control=control)
        outputs.append(y)
        stats.append(current)
    return outputs, state, stats
