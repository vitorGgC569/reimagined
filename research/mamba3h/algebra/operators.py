"""Isolated CPU algebra controls, NOT a Mamba-3 recurrence or integration."""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
import itertools
import torch
from torch import nn
torch.set_num_threads(2)

S5_PAIRS = tuple(itertools.combinations(range(5), 2))


def _cpu(*xs):
    if any(x.device.type != 'cpu' for x in xs if isinstance(x, torch.Tensor)):
        raise ValueError('CPU only')


def transition(h, u, alpha, diagonal=None):
    """Apply D first, then chronological normalized rank-one factors.

    h [B,D], u [B,R,D], alpha [B,R]. Zero directions are invalid.
    A = H_R ... H_1 D, H_j = I-alpha_j v_j v_j^T.
    """
    _cpu(h, u, alpha, diagonal)
    if h.ndim != 2 or u.ndim != 3 or u.shape[0] != h.shape[0] or u.shape[2] != h.shape[1]:
        raise ValueError('expected h[B,D], u[B,R,D]')
    if alpha.shape != u.shape[:2]:
        raise ValueError('expected alpha[B,R]')
    if not all(torch.isfinite(t).all() for t in (h, u, alpha)):
        raise ValueError('nonfinite input')
    norms = u.norm(dim=-1, keepdim=True)
    if (norms <= 1e-12).any():
        raise ValueError('zero direction')
    if diagonal is not None:
        if diagonal.shape not in (h.shape, (h.shape[1],)) or not torch.isfinite(diagonal).all():
            raise ValueError('bad diagonal')
        h = h * diagonal
    v = u / norms
    for j in range(u.shape[1]):
        h = h - alpha[:, j:j+1] * (h * v[:, j]).sum(-1, keepdim=True) * v[:, j]
    return h


def permutation_matrix(perm, dtype=torch.float64):
    if sorted(perm) != list(range(len(perm))):
        raise ValueError('not a permutation')
    return torch.eye(len(perm), dtype=dtype)[:, torch.tensor(perm)]


def s5_matrices(dtype=torch.float64):
    out = []
    for i, j in S5_PAIRS:
        u = torch.zeros(1, 1, 5, dtype=dtype)
        u[0, 0, i], u[0, 0, j] = 1., -1.
        # Rows of eye are column probes; transpose restores matrix convention.
        out.append(transition(torch.eye(5, dtype=dtype), u.expand(5, -1, -1),
                              torch.full((5, 1), 2., dtype=dtype)).T)
    return torch.stack(out)


def compose(late, early):
    """Affine chronological composition; accepts batched A and b."""
    a2, b2 = late
    a1, b1 = early
    return a2 @ a1, (a2 @ b1.unsqueeze(-1)).squeeze(-1) + b2


def _fold(ops, identity):
    acc = identity
    for op in ops:
        acc = compose(op, acc)
    return acc


def _tree(ops, identity):
    if not ops:
        return identity
    if len(ops) == 1:
        return ops[0]
    mid = len(ops) // 2
    return compose(_tree(ops[mid:], identity), _tree(ops[:mid], identity))


def scan(a, b, h0, method='sequential', chunk=4):
    """Inclusive affine states [T,B,D]; exact dense composition, no compression.

    Chunk builds each local prefix and composes with a running carry.
    Tree recursively composes each prefix (correctness reference, not fast scan).
    """
    _cpu(a, b, h0)
    if a.ndim != 4 or b.shape != a.shape[:3] or a.shape[-1] != a.shape[-2] or h0.shape != b.shape[1:]:
        raise ValueError('expected A[T,B,D,D], b[T,B,D], h0[B,D]')
    if method not in ('sequential', 'chunk', 'tree') or chunk < 1:
        raise ValueError('bad scan method/chunk')
    ops = list(zip(a.unbind(), b.unbind()))
    identity = (torch.eye(h0.shape[-1], dtype=h0.dtype).expand(h0.shape[0], -1, -1), torch.zeros_like(h0))
    if not ops:
        return h0.new_empty((0,) + h0.shape)
    states = []
    if method == 'sequential':
        h = h0
        for aa, bb in ops:
            h = (aa @ h.unsqueeze(-1)).squeeze(-1) + bb
            states.append(h)
    elif method == 'tree':
        for end in range(1, len(ops)+1):
            aa, bb = _tree(ops[:end], identity)
            states.append((aa @ h0.unsqueeze(-1)).squeeze(-1) + bb)
    else:
        carry = identity
        for start in range(0, len(ops), chunk):
            local = identity
            for op in ops[start:start+chunk]:
                local = compose(op, local)
                aa, bb = compose(local, carry)
                states.append((aa @ h0.unsqueeze(-1)).squeeze(-1) + bb)
            carry = compose(local, carry)
    return torch.stack(states)


def recompress_identity(a, rank):
    """Deliberately approximate I+rank-r residual; no closure guarantee."""
    if rank < 1 or rank > a.shape[-1]:
        raise ValueError('bad rank')
    eye = torch.eye(a.shape[-1], dtype=a.dtype)
    u, s, vh = torch.linalg.svd(a-eye)
    return eye + (u[..., :rank] * s[..., :rank].unsqueeze(-2)) @ vh[..., :rank, :]


class StructuredTransition(nn.Module):
    """Parameter-matched isolated NC / common-basis commuting adapter.

    Both arms reserve K*(R*D+R+D) learned scalars. Commuting maps directions
    to common-basis weights, so effective capacity/FLOPs are NOT identical.
    Every homogeneous operator is contractive: alpha in [0,2], |D|<=1.
    Additive x can still grow state norms. No native backbone is included.
    """
    def __init__(self, dim=8, rank=1, operations=10, commuting=False, seed=11, dtype=torch.float64):
        super().__init__()
        if rank not in (1, 2, 4) or dim < rank or operations < 1:
            raise ValueError('invalid dim/rank/operations')
        g = torch.Generator().manual_seed(seed)
        self.dim, self.rank, self.operations, self.commuting = dim, rank, operations, commuting
        self.u = nn.Parameter(torch.randn(operations, rank, dim, generator=g, dtype=dtype))
        self.alpha_raw = nn.Parameter(torch.zeros(operations, rank, dtype=dtype))
        self.diagonal_raw = nn.Parameter(torch.zeros(operations, dim, dtype=dtype))
        q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g, dtype=dtype))
        self.register_buffer('basis', q)

    def matrices(self, op_ids):
        _cpu(op_ids, self.u)
        if op_ids.ndim != 1 or op_ids.dtype != torch.long or (op_ids < 0).any() or (op_ids >= self.operations).any():
            raise ValueError('op_id must be legal int64[B]')
        u = self.u[op_ids]
        alpha = 2*torch.sigmoid(self.alpha_raw[op_ids])
        diagonal = .98+.02*torch.sigmoid(self.diagonal_raw[op_ids])
        if self.commuting:
            weights = torch.softmax(u, -1)
            eig = diagonal * torch.prod(1-alpha.unsqueeze(-1)*weights, dim=1)
            return (self.basis.unsqueeze(0)*eig.unsqueeze(-2)) @ self.basis.T
        eye = torch.eye(self.dim, dtype=self.u.dtype).expand(len(op_ids), -1, -1)
        # Each matrix column is a separate input batch to transition.
        states = eye.transpose(-1, -2).reshape(-1, self.dim)
        out = transition(states, u.repeat_interleave(self.dim, 0),
                         alpha.repeat_interleave(self.dim, 0), diagonal.repeat_interleave(self.dim, 0))
        return out.reshape(-1, self.dim, self.dim).transpose(-1, -2)

    def step(self, x, state, *, control=None):
        _cpu(x, state)
        if x.ndim != 2 or x.shape[1] != self.dim or not torch.isfinite(x).all():
            raise ValueError('expected finite x[B,D]')
        if control is None or set(control)-{'op_id', 'enabled'}:
            raise ValueError('only causal op_id and enabled accepted')
        if state is None:
            state = torch.zeros_like(x)
        if state.shape != x.shape or state.dtype != x.dtype or not torch.isfinite(state).all():
            raise ValueError('bad explicit state')
        enabled = control.get('enabled', True)
        if not isinstance(enabled, bool):
            raise ValueError('enabled must be bool')
        if not enabled:
            return x, state, self.stats(state, 0)
        ids = control.get('op_id')
        if not isinstance(ids, torch.Tensor) or ids.shape != (x.shape[0],):
            raise ValueError('expected op_id[B]')
        a = self.matrices(ids)
        h = (a @ state.unsqueeze(-1)).squeeze(-1) + x
        return h, h, self.stats(h, len(ids))

    def stats(self, state, operator_count):
        return dict(parameter_count=sum(p.numel() for p in self.parameters()),
                    parameter_bytes=sum(p.numel()*p.element_size() for p in self.parameters()),
                    state_bytes=state.numel()*state.element_size(),
                    persistent_buffer_bytes=self.basis.numel()*self.basis.element_size(),
                    cache_bytes=0, live_operator_bytes=operator_count*self.dim*self.dim*self.u.element_size(),
                    workspace_bytes_measured=False, operator_count=operator_count, rank=self.rank,
                    commuting=self.commuting, scope='isolated_algebra_adapter')
