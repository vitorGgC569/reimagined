"""P1 naming/check adapter over frozen P0. Contains no token values or labels.

Nonnegative entity filtering is SYMBOLIC ADDRESS, even when address=None.
Literal WRA indices can be redundant with that symbolic address. Q/K scores,
softmax, stored values and output/readout remain the frozen module's functions.
"""
from dataclasses import dataclass
import torch
from research.mamba3h.memory.slots import Control, OracleRoutes, route_oracle


@dataclass(frozen=True)
class Operation:
    kind: str
    entity: int


def causal_control(mode, operations, state, *, symbolic_address=False, literal_address=False):
    """Route from CURRENT opcode/ID and PAST occupied/ID metadata only.

    W controls WRITE; WR controls WRITE/READ; WRA also literal ADDRESS. This P1
    adapter deliberately does not use the P0 WR retention extension. Absent
    literal indices do not imply learned addressing if symbolic_address=True.
    A caller requesting literal ADDRESS must have past entity-tagged slots;
    anonymous caches need a separate causal directory, whose bytes are extra.
    """
    if mode not in ('W','WR','WRA','store_all'):
        raise ValueError('unknown routing mode')
    if literal_address and mode!='WRA':
        raise ValueError('literal address requires WRA')
    if len(operations)!=state.clock.numel():
        raise ValueError('batch mismatch')
    for op in operations:
        if type(op) is not Operation or op.kind not in ('WRITE','QUERY','NOOP','REVOKE'):
            raise TypeError('current Operation metadata required')
        if type(op.entity) is not int or not 0<=op.entity<4:
            raise ValueError('P1 has four entity IDs')
    batch=len(operations)
    write=torch.tensor([op.kind=='WRITE' for op in operations],dtype=torch.bool)
    read=torch.tensor([op.kind=='QUERY' for op in operations],dtype=torch.bool)
    entity=torch.tensor([op.entity if symbolic_address else -1 for op in operations],dtype=torch.long)
    revoke=torch.tensor([op.entity if op.kind=='REVOKE' else -1 for op in operations],dtype=torch.long)
    base=Control(torch.zeros(batch,dtype=torch.bool),torch.zeros(batch,dtype=torch.bool),entity,revoke)
    if mode=='store_all':
        # Store-all is only a write policy; READ uses the current QUERY opcode.
        return route_oracle(mode,Control(base.write,read,entity,revoke))
    oracle=OracleRoutes(write=write,read=read if mode in ('WR','WRA') else None)
    if literal_address:
        # k is 2 in this fixed P1 check. IDs/timestamps are prior metadata, not
        # future contents, projected values, targets or a content-derived score.
        indices=torch.full((batch,2),-1,dtype=torch.long)
        for b,op in enumerate(operations):
            if op.kind!='QUERY':
                continue
            hits=torch.nonzero(state.occupied[b] & (state.entities[b]==op.entity),as_tuple=False).flatten()
            if hits.numel()>1:
                raise ValueError('symbolic entity has multiple evidence slots')
            if hits.numel():
                if state.timestamps[b,hits[0]]>=state.clock[b]:
                    raise ValueError('future metadata is not an address oracle')
                indices[b,0]=hits[0]
        oracle=OracleRoutes(write=oracle.write,read=oracle.read,address=indices)
    return route_oracle(mode,base,oracle)
