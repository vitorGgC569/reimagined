"""Optional isolated components; lazy import lets runners cap threads first."""
__all__ = ["CausalSlotMemory", "SparseRetrievalAttention", "SlotState", "Control", "OracleRoutes", "route_oracle"]


def __getattr__(name):
    if name in __all__:
        from . import slots
        return getattr(slots, name)
    raise AttributeError(name)
