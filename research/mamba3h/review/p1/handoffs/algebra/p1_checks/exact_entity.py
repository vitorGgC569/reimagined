"""Isolated exact known-operator control, NOT actual-native/NCE2E.

Only a current observed event enters step. No solver, target or future input.
Entity addressing, SET decoding, operators and readout are fixed, not learned.
"""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
import itertools
import torch
from torch import nn
torch.set_num_threads(2)
PAIRS=tuple(itertools.combinations(range(5),2))
FIELDS={'kind','entity','operator','value'}


def validate_event(event, entities):
    if not isinstance(event,dict) or set(event)!=FIELDS:
        raise ValueError('only current observed event fields allowed')
    kind=event['kind']
    if kind not in ('SET','OP','QUERY','REVOKE','NOOP','DISTRACTOR'):
        raise ValueError('INST S5 event kind required')
    if type(event['entity']) is not int or not 0<=event['entity']<entities:
        raise ValueError('entity outside configured universe')
    if kind=='OP':
        if type(event['operator']) is not int or not 0<=event['operator']<10:
            raise ValueError('current OP id in [0,10) required')
    elif event['operator'] is not None:
        raise ValueError('operator belongs only to OP token')
    if kind in ('SET','DISTRACTOR'):
        limit=5 if kind=='SET' else 125
        if type(event['value']) is not int or not 0<=event['value']<limit:
            raise ValueError('observed value outside token space')
    elif event['value'] is not None:
        raise ValueError('only observed SET/DISTRACTOR token carries a value')


class ExactEntityControl(nn.Module):
    """state[B,slots,5]; per_entity slots=E, global slots=1.

    All-zero is revoked/uninitialized. A fixed six-class linear readout maps
    onehot state to point0..4 and zero state to compact undefined class5.
    Global control deliberately shares every entity's stream in one slot.
    """
    def __init__(self, entities=3, per_entity=True, dtype=torch.float64):
        super().__init__()
        if type(entities) is not int or not 1<=entities<=64 or type(per_entity) is not bool:
            raise ValueError('entities [1,64], boolean per_entity required')
        if dtype not in (torch.float32,torch.float64):
            raise ValueError('CPU FP32/FP64 required')
        self.entities,self.per_entity=entities,per_entity
        self.slots=entities if per_entity else 1
        directions=torch.zeros(10,5,dtype=dtype)
        for k,(i,j) in enumerate(PAIRS):
            directions[k,i]=1.;directions[k,j]=-1.
        directions=directions/directions.norm(dim=-1,keepdim=True)
        self.register_buffer('directions',directions)
        self.register_buffer('readout_weight',torch.cat([torch.eye(5,dtype=dtype),-torch.ones(1,5,dtype=dtype)]))
        self.register_buffer('readout_bias',torch.tensor([0,0,0,0,0,1],dtype=dtype))

    def initial_state(self,batch=1):
        if type(batch) is not int or batch<1: raise ValueError('positive batch required')
        return self.directions.new_zeros(batch,self.slots,5)

    def readout(self,h):
        return h@self.readout_weight.T+self.readout_bias

    def step(self, events, state=None):
        """Returns logits[B,6], next explicit state, counters.

        QUERY is nonmutating. Non-QUERY logits are ignored by evaluation.
        A list contains exactly one current event per independent sample.
        """
        if not isinstance(events,(list,tuple)) or not events: raise ValueError('nonempty current-event batch required')
        for event in events: validate_event(event,self.entities)
        if self.directions.device.type!='cpu': raise ValueError('CPU only')
        if state is None: state=self.initial_state(len(events))
        if state.device.type!='cpu' or state.dtype!=self.directions.dtype or state.shape!=(len(events),self.slots,5):
            raise ValueError('state shape/device/dtype mismatch')
        if not torch.isfinite(state).all(): raise ValueError('nonfinite state')
        next_state=state.clone();selected=[];counts=dict(SET=0,OP=0,QUERY=0,REVOKE=0,NOOP=0,DISTRACTOR=0)
        for batch,event in enumerate(events):
            slot=event['entity'] if self.per_entity else 0
            kind=event['kind'];counts[kind]+=1
            if kind=='SET':
                # Only the value present on this observed SET token is decoded.
                next_state[batch,slot]=torch.nn.functional.one_hot(torch.tensor(event['value']),5).to(state.dtype)
            elif kind=='OP':
                u=self.directions[event['operator']]
                h=state[batch,slot]
                next_state[batch,slot]=h-2*(h*u).sum()*u
            elif kind=='REVOKE':
                next_state[batch,slot]=torch.zeros(5,dtype=state.dtype)
            selected.append(next_state[batch,slot])
        logits=self.readout(torch.stack(selected))
        return logits,next_state,dict(counts=counts,parameter_count=0,
            fixed_buffer_bytes=sum(b.numel()*b.element_size() for b in self.buffers()),
            persistent_state_bytes=next_state.numel()*next_state.element_size(),
            active_state_dimensions_per_sample=self.slots*5,cache_bytes=0,
            retrieval_similarity_operations=0,scope='isolated_exact_known_operator_control_NOT_actualnative_NCE2E')


def run_events(events,model):
    """Only events supplied here; returns compact classes5=undefined, None off-query."""
    state=None;predictions=[];states=[];stats=[]
    for event in events:
        logits,state,counters=model.step([event],state)
        predictions.append(int(logits.argmax(-1)[0]) if event['kind']=='QUERY' else None)
        states.append(state.clone());stats.append(counters)
    return predictions,states,stats
