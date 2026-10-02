"""Independent query-local replay solvers.

Never imports generation.py, its online labeler, permutation table or group
operators. MQAR searches backwards; INST replays only a query's entity suffix;
S5 group uses inverse maps plus an independent Lehmer encoder.
"""

from .schema import IGNORE, UNDEFINED, model_view


def _pair(op):
    n = 0
    for a in range(5):
        for b in range(a + 1, 5):
            if n == op:
                return a, b
            n += 1
    raise ValueError("Unknown transposition")


def _unrank(rank):
    pool, result = list(range(5)), []
    for factor in (24, 6, 2, 1, 1):
        digit, rank = divmod(rank, factor)
        result.append(pool.pop(digit))
    return result


def _rank(perm):
    total = 0
    for i, factor in enumerate((24, 6, 2, 1, 1)):
        total += sum(perm[j] < perm[i] for j in range(i + 1, 5)) * factor
    return total


def _shift(op):
    # Independent explicit construction, not the generator's lookup table.
    if op < 3:
        return [int(axis == op) for axis in range(3)]
    if op < 6:
        return [-int(axis == op - 3) for axis in range(3)]
    return {6: [1, 1, 0], 7: [0, 1, 1], 8: [1, 0, 1], 9: [1, 1, 1]}[op]


def _query(task, group, prefix, entity):
    selected = [ev for ev in prefix if ev["entity"] == entity]
    anchor, raw = -1, 0 if task == "group" else None
    for j in range(len(selected) - 1, -1, -1):
        ev = selected[j]
        if ev["kind"] in ("WRITE", "SET", "REVOKE"):
            anchor = j
            raw = None if ev["kind"] == "REVOKE" else ev["value"]
            break
    if raw is None:
        return UNDEFINED
    if task == "mqar":
        return raw
    ops = [ev["operator"] for ev in selected[anchor + 1:] if ev["kind"] == "OP"]
    if group == "abelian":
        digits = [raw // 25, raw // 5 % 5, raw % 5]
        # Sum the whole query suffix, then reduce once rather than mutable replay.
        moves = [_shift(op) for op in ops]
        values = [(digits[axis] + sum(v[axis] for v in moves)) % 5 for axis in range(3)]
        return 25 * values[0] + 5 * values[1] + values[2]
    if task == "inst":
        point = raw
        for op in ops:
            a, b = _pair(op)
            if point in (a, b):
                point = a + b - point
        return point
    permutation = _unrank(raw)
    inverse = [permutation.index(point) for point in range(5)]
    for op in ops:
        a, b = _pair(op)
        inverse[a], inverse[b] = inverse[b], inverse[a]
    return _rank([inverse.index(point) for point in range(5)])


def solve(episode):
    """Return length-L targets; read no targets, RNG state, metadata or future."""
    view = model_view(episode)
    return [_query(view["task"], view["group"], view["events"][:t], ev["entity"])
            if ev["kind"] == "QUERY" else IGNORE for t, ev in enumerate(view["events"])]


def query_trace(episode):
    labels = solve(episode)
    return [{"event_index": t, "answer": None if v == UNDEFINED else v,
             "defined": v != UNDEFINED, "class_id": v}
            for t, v in enumerate(labels) if v != IGNORE]
