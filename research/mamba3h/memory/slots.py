"""Differentiable, explicit-state CPU memory. Controls carry routing metadata only.

Read precedes the current write. Revoke, retain and overwrite invalidation precede
read, so stale evidence cannot contribute even on combined read/write steps.
Top-k is exact with stable lower-slot-index tie breaking. Oracle addressing changes
indices only: Q/K logits, values, softmax and output projection stay model-derived.
"""
from dataclasses import dataclass, replace
from typing import Optional
import math
import torch
from torch import Tensor, nn


COUNTERS = ("read_requests", "reads", "similarity_evaluations", "similarity_madds",
            "topk_calls", "topk_candidates", "address_candidates", "value_reads",
            "writes", "evictions", "overwrites", "revocations", "retention_drops",
            "projection_madds", "output_madds", "attention_madds", "candidate_slot_checks")


@dataclass(frozen=True)
class Control:
    write: Tensor
    read: Tensor
    entity: Optional[Tensor] = None
    revoke_entity: Optional[Tensor] = None
    retain: Optional[Tensor] = None
    address: Optional[Tensor] = None


@dataclass(frozen=True)
class OracleRoutes:
    """No content, target, score, logit, value or readout field exists."""
    write: Optional[Tensor] = None
    read: Optional[Tensor] = None
    retain: Optional[Tensor] = None
    address: Optional[Tensor] = None


def route_oracle(mode: str, base: Control, oracle: Optional[OracleRoutes] = None) -> Control:
    """W=write; WR=write/read/retention; WRA also discrete address; store_all=write.

    The caller must derive oracle metadata solely from the visible prefix.
    Shape, address validity and causal state checks happen in step().
    """
    if type(base) is not Control or (oracle is not None and type(oracle) is not OracleRoutes):
        raise TypeError("Only routing dataclasses are accepted")
    if mode not in ("none", "W", "WR", "WRA", "store_all"):
        raise ValueError("Unknown routing mode")
    allowed = {"none": (), "W": ("write",), "WR": ("write", "read", "retain"),
               "WRA": ("write", "read", "retain", "address"), "store_all": ()}[mode]
    oracle = oracle or OracleRoutes()
    updates = {}
    for name in ("write", "read", "retain", "address"):
        value = getattr(oracle, name)
        if value is not None:
            if name not in allowed:
                raise ValueError(f"{mode} cannot oracle-control {name}")
            updates[name] = value
    if mode == "store_all":
        updates["write"] = torch.ones_like(base.write, dtype=torch.bool)
    return replace(base, **updates)


@dataclass(frozen=True)
class SlotState:
    keys: Tensor
    values: Tensor
    occupied: Tensor
    entities: Tensor
    timestamps: Tensor
    clock: Tensor
    counters: Tensor

    def detach(self):
        return SlotState(*(getattr(self, f).detach().clone() for f in self.__dataclass_fields__))

    def tensor_bytes(self):
        return sum(getattr(self, f).numel() * getattr(self, f).element_size()
                   for f in self.__dataclass_fields__)


class CausalSlotMemory(nn.Module):
    """Fixed reserved capacity, chronological FIFO eviction, entity-scoped reads.

    Anonymous entity=-1 searches all active slots. Nonnegative entities search only
    their own evidence. Explicit routing is recommended for operation benchmarks;
    control=None uses thresholded learned gates (discrete gates have no STE).
    Capacity and top_k are budgets, not hints. State is caller-owned and immutable.
    """
    def __init__(self, dim: int, capacity: int, top_k: int = 1, key_dim: Optional[int] = None):
        super().__init__()
        key_dim = dim if key_dim is None else key_dim
        if any(type(n) is not int or n <= 0 for n in (dim, capacity, top_k, key_dim)):
            raise ValueError("dimensions, capacity and top_k must be positive integers")
        if top_k > capacity:
            raise ValueError("top_k exceeds capacity")
        self.dim, self.capacity, self.top_k, self.key_dim = dim, capacity, top_k, key_dim
        self.query = nn.Linear(dim, key_dim, bias=False)
        self.key = nn.Linear(dim, key_dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.gates = nn.Linear(dim, 3)

    def initial_state(self, batch: int, *, dtype=None) -> SlotState:
        if type(batch) is not int or batch <= 0:
            raise ValueError("positive batch required")
        dtype = dtype or self.query.weight.dtype
        if self.query.weight.device.type != "cpu":
            raise ValueError("P0 is CPU-only")
        return SlotState(torch.zeros(batch, self.capacity, self.key_dim, dtype=dtype),
                         torch.zeros(batch, self.capacity, self.dim, dtype=dtype),
                         torch.zeros(batch, self.capacity, dtype=torch.bool),
                         torch.full((batch, self.capacity), -1, dtype=torch.long),
                         torch.full((batch, self.capacity), -1, dtype=torch.long),
                         torch.zeros(batch, dtype=torch.long),
                         torch.zeros(batch, len(COUNTERS), dtype=torch.long))

    def learned_control(self, x: Tensor, state: SlotState) -> Control:
        decisions = self.gates(x) >= 0
        return Control(decisions[:, 0], decisions[:, 1],
                       retain=decisions[:, 2, None].expand_as(state.occupied))

    def _validate(self, x, state, control):
        if type(state) is not SlotState or type(control) is not Control:
            raise TypeError("Expected explicit SlotState and routing-only Control")
        if x.device.type != "cpu" or x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError("x must be CPU [B,D]")
        if x.dtype != self.query.weight.dtype or not torch.isfinite(x).all():
            raise ValueError("x must be finite and match model dtype")
        b = x.shape[0]
        expected = {"keys": ((b,self.capacity,self.key_dim), x.dtype),
                    "values": ((b,self.capacity,self.dim), x.dtype),
                    "occupied": ((b,self.capacity), torch.bool),
                    "entities": ((b,self.capacity), torch.long),
                    "timestamps": ((b,self.capacity), torch.long),
                    "clock": ((b,), torch.long),
                    "counters": ((b,len(COUNTERS)), torch.long)}
        for name, (shape, dtype) in expected.items():
            t = getattr(state, name)
            if not isinstance(t, Tensor) or t.device.type != "cpu" or tuple(t.shape) != shape or t.dtype != dtype:
                raise ValueError(f"Invalid state {name}")
        if not torch.isfinite(state.keys).all() or not torch.isfinite(state.values).all():
            raise ValueError("nonfinite state")
        if (state.clock < 0).any() or (state.counters < 0).any():
            raise ValueError("negative state metadata")
        if (state.occupied & ((state.timestamps < 0) | (state.timestamps >= state.clock[:, None]))).any():
            raise ValueError("future or invalid active timestamp")
        for name, shape, dtype in (("write", (b,), torch.bool), ("read", (b,), torch.bool),
                                   ("entity", (b,), torch.long), ("revoke_entity", (b,), torch.long),
                                   ("retain", (b,self.capacity), torch.bool),
                                   ("address", (b,self.top_k), torch.long)):
            t = getattr(control, name)
            if t is None and name not in ("write", "read"):
                continue
            if not isinstance(t, Tensor) or t.device.type != "cpu" or tuple(t.shape) != shape or t.dtype != dtype:
                raise ValueError(f"Invalid control {name}")
        for name in ("entity", "revoke_entity"):
            t = getattr(control, name)
            if t is not None and (t < -1).any():
                raise ValueError("entity IDs must be -1 or nonnegative")
        if control.address is not None and ((control.address < -1) | (control.address >= self.capacity)).any():
            raise ValueError("address outside budget")

    def step(self, x: Tensor, state: SlotState, control: Optional[Control] = None):
        if control is None:
            control = self.learned_control(x, state)
        self._validate(x, state, control)
        # All three projections use x, never y or retrieved contents.
        q, k, v = self.query(x), self.key(x), self.value(x)
        bsz = x.shape[0]
        entity = control.entity if control.entity is not None else torch.full((bsz,), -1, dtype=torch.long)
        revoke = control.revoke_entity if control.revoke_entity is not None else torch.full_like(entity, -1)
        revoked = state.occupied & (revoke[:, None] >= 0) & (state.entities == revoke[:, None])
        overwritten = state.occupied & control.write[:, None] & (entity[:, None] >= 0) & (state.entities == entity[:, None]) & ~revoked
        dropped = state.occupied & ~revoked & ~overwritten
        dropped = dropped & ~control.retain if control.retain is not None else torch.zeros_like(dropped)
        removed = revoked | overwritten | dropped
        occupied = state.occupied & ~removed
        keys = state.keys.masked_fill(removed[..., None], 0)
        values = state.values.masked_fill(removed[..., None], 0)
        entities = state.entities.masked_fill(removed, -1)
        stamps = state.timestamps.masked_fill(removed, -1)
        counts = torch.zeros_like(state.counters)
        col = {name:i for i,name in enumerate(COUNTERS)}
        counts[:,col["projection_madds"]] = 2*self.dim*self.key_dim + self.dim*self.dim
        counts[:,col["output_madds"]] = self.dim*self.dim
        counts[:,col["revocations"]] = revoked.sum(1)
        counts[:,col["overwrites"]] = overwritten.sum(1)
        counts[:,col["retention_drops"]] = dropped.sum(1)
        counts[:,col["read_requests"]] = control.read.to(torch.long)
        retrieved = []
        peak_search_bytes = 0
        selected_indices = []
        for b in range(bsz):
            # Branch BEFORE candidate construction/similarity/top-k.
            if not bool(control.read[b]):
                retrieved.append(torch.zeros_like(x[b]))
                selected_indices.append([])
                continue
            counts[b,col["reads"]] += 1
            counts[b,col["candidate_slot_checks"]] += self.capacity
            eligible = occupied[b] & ((entities[b] == entity[b]) if entity[b] >= 0 else True)
            indices = torch.nonzero(eligible, as_tuple=False).flatten()
            if control.address is not None:
                chosen = control.address[b]
                chosen = chosen[chosen >= 0]
                counts[b,col["address_candidates"]] += chosen.numel()
                if chosen.unique().numel() != chosen.numel() or (chosen.numel() and not eligible[chosen].all()):
                    raise ValueError("duplicate, stale, future or cross-entity oracle address")
                indices = chosen
            if indices.numel() == 0:
                retrieved.append(torch.zeros_like(x[b]))
                selected_indices.append([])
                continue
            scores = keys[b].index_select(0, indices) @ q[b] / math.sqrt(self.key_dim)
            counts[b,col["similarity_evaluations"]] += indices.numel()
            counts[b,col["similarity_madds"]] += indices.numel() * self.key_dim
            if control.address is None:
                counts[b,col["topk_calls"]] += 1
                counts[b,col["topk_candidates"]] += indices.numel()
                # Stable exact order: scores descending, slot index ascending ties.
                chosen_order = torch.argsort(scores, descending=True, stable=True)[:self.top_k]
                chosen = indices[chosen_order]
                selected_scores = scores[chosen_order]
            else:
                chosen, selected_scores = indices, scores
            weights = torch.softmax(selected_scores, dim=0)
            read_values = values[b].index_select(0, chosen)
            retrieved.append(weights @ read_values)
            counts[b,col["value_reads"]] += chosen.numel()
            counts[b,col["attention_madds"]] += chosen.numel()*self.dim
            selected_indices.append(chosen.tolist())
            # Explicit tensor workspace estimate; not allocator/profiler measurement.
            workspace = (indices.numel()*8 + scores.numel()*scores.element_size()
                         + indices.numel()*self.key_dim*x.element_size()
                         + chosen.numel()*(8+2*x.element_size()+self.dim*x.element_size()))
            peak_search_bytes = max(peak_search_bytes, workspace)
        r = torch.stack(retrieved)
        y = x + self.output(r)
        # Clones preserve caller state, and avoid mutating tensors saved by autograd.
        keys, values, occupied, entities, stamps = (t.clone() for t in (keys,values,occupied,entities,stamps))
        for b in range(bsz):
            if not bool(control.write[b]):
                continue
            free = torch.nonzero(~occupied[b], as_tuple=False).flatten()
            if free.numel():
                slot = int(free[0])
            else:
                slot = int(torch.argmin(stamps[b]))
                counts[b,col["evictions"]] += 1
            keys[b,slot], values[b,slot] = k[b], v[b]
            occupied[b,slot], entities[b,slot], stamps[b,slot] = True, entity[b], state.clock[b]
            counts[b,col["writes"]] += 1
        next_state = SlotState(keys,values,occupied,entities,stamps,state.clock+1,state.counters+counts)
        stats = {name:int(counts[:,i].sum()) for i,name in enumerate(COUNTERS)}
        stats.update(occupancy=occupied.sum(1).tolist(), selected_indices=selected_indices,
                     parameter_bytes=sum(p.numel()*p.element_size() for p in self.parameters()),
                     state_bytes=next_state.tensor_bytes(),
                     cache_bytes_estimate=(q.numel()+k.numel()+v.numel()+r.numel())*x.element_size()+peak_search_bytes,
                     cache_bytes_kind="tensor_workspace_estimate_not_allocator_peak",
                     component_label="isolated_memory_component",
                     backend=type(self).__name__)
        return y, next_state, stats

    def forward(self, x, state, *, control=None):
        return self.step(x,state,control)


class SparseRetrievalAttention(CausalSlotMemory):
    """Bounded FIFO prefix K/V cache with exact sparse attention.

    Shares the same Q/K/V/output module and operation accounting. Use store_all
    routing for the MA control; cache retains last C tokens (anonymous entity=-1).
    This is an attention control, not a distinct recurrence/native Mamba baseline.
    """
