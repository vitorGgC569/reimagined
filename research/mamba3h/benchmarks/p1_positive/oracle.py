"""Independent query-local backward replay; never imports the generator."""

from .schema import IGNORE, model_view


class UnboundQuery(ValueError):
    pass


def solve(ep):
    view = model_view(ep)
    targets = [IGNORE] * len(view["events"])
    for t, current in enumerate(view["events"]):
        if current["kind"] != "QUERY":
            continue
        for previous in reversed(view["events"][:t]):
            if previous["kind"] == "WRITE" and previous["entity"] == current["entity"]:
                targets[t] = previous["value"]
                break
        else:
            raise UnboundQuery(f"QUERY at {t} has no causal WRITE for entity {current['entity']}")
    return targets
