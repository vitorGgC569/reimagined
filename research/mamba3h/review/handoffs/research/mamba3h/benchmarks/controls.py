"""Deliberately weak diagnostic predictors, never learned/model baselines."""

from .schema import IGNORE, UNDEFINED, model_view


def predict_undefined(episode):
    return [UNDEFINED if query else IGNORE for query in model_view(episode)["query_mask"]]


def predict_first_write(episode):
    """MQAR stale-memory control: deliberately ignores overwrite and revocation."""
    view = model_view(episode)
    if view["task"] != "mqar":
        raise ValueError("first-write control is MQAR only")
    first, labels = {}, []
    for ev in view["events"]:
        if ev["kind"] == "WRITE":
            first.setdefault(ev["entity"], ev["value"])
        labels.append(first.get(ev["entity"], UNDEFINED) if ev["kind"] == "QUERY" else IGNORE)
    return labels


def predict_reversed_order(episode):
    """Noncausal-to-operation-order diagnostic (still past-only, never targets).

    Reverse the latest same-entity operation suffix separately at each query.
    Abelian answers should survive; noncommuting answers need not.
    """
    from .oracle import solve
    view = model_view(episode)
    if view["task"] == "mqar":
        raise ValueError("Order control requires group/INST")
    result = [IGNORE] * len(view["events"])
    for t, ev in enumerate(view["events"]):
        if ev["kind"] != "QUERY":
            continue
        prefix = [dict(e) for e in view["events"][:t + 1]]
        indices = []
        for j in range(t - 1, -1, -1):
            e = prefix[j]
            if e["entity"] != ev["entity"]:
                continue
            if e["kind"] in ("SET", "REVOKE"):
                break
            if e["kind"] == "OP":
                indices.append(j)
        operators = [prefix[j]["operator"] for j in reversed(indices)]
        for j, op in zip(indices, operators):
            prefix[j]["operator"] = op
        result[t] = solve(dict(view, events=prefix))[t]
    return result
