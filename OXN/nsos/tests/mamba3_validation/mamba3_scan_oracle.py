"""Independent NumPy FP64 oracle; never imports production math or GPU libraries.

Affine pairs encode H -> a*H+u. Composition is chronological (later after earlier).
Checkpoint replay here simulates bounded LDS state storage on CPU, not GPU execution.
"""
from dataclasses import dataclass
import math
import numpy as np

PIN = 'e9594ce1c732d97440f0332fdc43170a2294dbfa'


def sigmoid(x):
    return np.exp(-np.logaddexp(0., -np.asarray(x)))


def affine_prefix(alpha, forcing, initial, chunk=32):
    """Inclusive doubling scan within tiles, carrying the previous tile's state.

    No log-cumsum division: exact zero/underflowed decay is well-defined.
    """
    if chunk <= 0:
        raise ValueError('positive chunk required')
    out = np.empty_like(forcing)
    carry = initial.copy()
    for start in range(0, len(alpha), chunk):
        stop = min(start + chunk, len(alpha))
        a = alpha[start:stop].copy()
        u = forcing[start:stop].copy()
        offset = 1
        while offset < len(a):
            olda, oldu = a.copy(), u.copy()
            a[offset:] = olda[offset:] * olda[:-offset]
            u[offset:] = oldu[offset:] + olda[offset:, None, None] * oldu[:-offset]
            offset *= 2
        out[start:stop] = u + a[:, None, None] * carry
        carry = out[stop - 1].copy()
    return out


def quadratic_states(alpha, forcing, initial):
    """Independent dense causal expansion, O(S^2); no recurrence helpers."""
    out = np.zeros_like(forcing)
    for t in range(len(alpha)):
        tail = 1.
        for j in range(t, -1, -1):
            out[t] += tail * forcing[j]
            tail *= alpha[j]
        out[t] += tail * initial
    return out


def forcing_terms(beta, gamma, k, v, k0, v0):
    # k [T,R,N], v [T,R,P], shared state [P,N]; rank is contracted before scan.
    prevk = np.concatenate((k0[None], k[:-1]), axis=0)
    prevv = np.concatenate((v0[None], v[:-1]), axis=0)
    return (beta[:, None, None] * np.einsum('trn,trp->tpn', prevk, prevv)
            + gamma[:, None, None] * np.einsum('trn,trp->tpn', k, v))


def operand_forward(alpha, beta, gamma, q, k, v, h0, k0, v0, method='scan', chunk=32):
    u = forcing_terms(beta, gamma, k, v, k0, v0)
    if method == 'quadratic':
        states = quadratic_states(alpha, u, h0)
    elif method == 'scan':
        states = affine_prefix(alpha, u, h0, chunk)
    elif method == 'serial':
        states = np.empty_like(u)
        state = h0.copy()
        for t in range(len(alpha)):
            state = alpha[t] * state + u[t]
            states[t] = state
    else:
        raise ValueError(method)
    return np.einsum('tpn,trn->trp', states, q), states


def operand_vjp(alpha, beta, gamma, q, k, v, h0, k0, v0, dy,
                seed_h, seed_k, seed_v, chunk=None):
    """Analytic reverse VJP; optionally retain boundaries and replay one tile.

    All operand and boundary adjoints included. Parameter/preprocessing chains
    are checked independently by full-layer directional derivatives in tests.
    """
    T = len(alpha)
    if chunk is not None and chunk <= 0:
        raise ValueError('positive chunk required')
    u = forcing_terms(beta, gamma, k, v, k0, v0)
    tile = max(T, 1) if chunk is None else chunk
    boundaries = [h0.copy()]
    state = h0.copy()
    full = [state.copy()] if chunk is None else None
    for t in range(T):
        state = alpha[t] * state + u[t]
        if full is not None:
            full.append(state.copy())
        if (t + 1) % tile == 0 or t + 1 == T:
            boundaries.append(state.copy())
    grads = {name: np.zeros_like(val) for name, val in
             [('alpha', alpha), ('beta', beta), ('gamma', gamma), ('q', q),
              ('k', k), ('v', v), ('h0', h0), ('k0', k0), ('v0', v0)]}
    dh = seed_h.copy()
    if T:
        grads['k'][-1] += seed_k
        grads['v'][-1] += seed_v
    else:
        grads['k0'] += seed_k
        grads['v0'] += seed_v
    for start in reversed(range(0, T, tile)):
        end = min(start + tile, T)
        if full is not None:
            local = full[start:end + 1]
        else:
            local = [boundaries[start // tile].copy()]
            for t in range(start, end):
                local.append(alpha[t] * local[-1] + u[t])
        for t in reversed(range(start, end)):
            ht, hp = local[t-start+1], local[t-start]
            grads['q'][t] += np.einsum('rp,pn->rn', dy[t], ht)
            dh += np.einsum('rp,rn->pn', dy[t], q[t])
            pk, pv = (k[t-1], v[t-1]) if t else (k0, v0)
            grads['alpha'][t] = np.sum(dh * hp)
            grads['beta'][t] = np.sum(dh * np.einsum('rn,rp->pn', pk, pv))
            grads['gamma'][t] = np.sum(dh * np.einsum('rn,rp->pn', k[t], v[t]))
            grads['k'][t] += gamma[t] * np.einsum('pn,rp->rn', dh, v[t])
            grads['v'][t] += gamma[t] * np.einsum('pn,rn->rp', dh, k[t])
            gpk = beta[t] * np.einsum('pn,rp->rn', dh, pv)
            gpv = beta[t] * np.einsum('pn,rn->rp', dh, pk)
            if t:
                grads['k'][t-1] += gpk
                grads['v'][t-1] += gpv
            else:
                grads['k0'] += gpk
                grads['v0'] += gpv
            dh *= alpha[t]
    grads['h0'] = dh
    # Only state storage estimate. q/k/v/u/dy and reduction buffers are excluded.
    state_elements = h0.size * ((T + 1) if chunk is None else
                               len(boundaries) + min(tile, T) + 1)
    return grads, state_elements


@dataclass(frozen=True)
class Geometry:
    heads: int = 2
    groups: int = 1
    head_dim: int = 4
    state_dim: int = 128
    rank: int = 1
    mimo: bool = False
    norm: bool = False
    rope: float = .5
    eps: float = 1e-5
    a_floor: float = 1e-4

    @property
    def pairs(self):
        return int(self.state_dim * self.rope / 2)


def zero_state(g, batch):
    return dict(phase=np.zeros((batch,g.heads,g.pairs)),
                ssm=np.zeros((batch,g.heads,g.head_dim,g.state_dim)),
                k=np.zeros((batch,g.heads,g.rank,g.state_dim)),
                v=np.zeros((batch,g.heads,g.head_dim)))


def block_forward(x, weights, initial, valid, g, method='scan', chunk=32, projected=False):
    """Full dense layer oracle in canonical registry order; separate raw V state.

    Includes projection, FP64 RMS, affine bias, phase, trapezoid, skip D, rank
    projections, optional per-rank output RMS, SiLU gate and output projection.
    Padded inputs are never read. This is mathematical FP64, not FP32 emulation.
    """
    B,S,_ = x.shape
    H,P,N,R,G,A = g.heads,g.head_dim,g.state_dim,g.rank,g.groups,g.pairs
    I = H*P
    iw,ow,bn,cn,dtbias,bbias,cbias,D = weights[:8]
    if g.mimo:
        px,pz,po = [np.asarray(w).reshape(H,R,P) for w in weights[8:11]]
        norm = weights[11] if g.norm else np.ones(I)
    else:
        px = pz = po = np.ones((H,1,P))
        norm = weights[8] if g.norm else np.ones(I)
    bbias,cbias = bbias.reshape(H,R,N),cbias.reshape(H,R,N)
    output = np.zeros((B,S,I))
    final = {name: val.copy() for name,val in initial.items()}
    for b in range(B):
        L = int(valid[b])
        if not 0 <= L <= S:
            raise ValueError('prefix outside sequence')
        if not L:
            continue
        raw = np.asarray(x[b,:L],dtype=np.float64)
        if not projected:
            raw = raw @ iw.T
        z,vr,k,q,dtr,ar,trap,angle = np.split(raw,
            np.cumsum([I,I,R*G*N,R*G*N,H,H,H]),axis=-1)
        k,q = k.reshape(L,R,G,N),q.reshape(L,R,G,N)
        dt = np.logaddexp(0.,dtr+dtbias)
        heaviside = np.empty_like(ar)
        positive = ar>=0
        heaviside[positive] = 1+ar[positive]
        heaviside[~positive] = 1/(1-ar[~positive])
        a = -np.maximum(heaviside,g.a_floor)
        alpha = np.exp(a*dt)
        lam = sigmoid(trap)
        beta,gamma = (1-lam)*dt*alpha,lam*dt
        for h in range(H):
            phase = (initial['phase'][b,h] + np.cumsum(
                math.pi*np.tanh(angle)*dt[:,h,None],axis=0)) % (2*math.pi)
            group = h//(H//G)
            rotated = []
            for operand,scale,bias in [(q,cn,cbias),(k,bn,bbias)]:
                bc = operand[:,:,group]
                bc = bc/np.sqrt(np.mean(bc*bc,axis=-1,keepdims=True)+g.eps)*scale+bias[h]
                left = np.arange(A) if g.mimo else 2*np.arange(A)
                right = left+N//2 if g.mimo else left+1
                u,v = bc[...,left].copy(),bc[...,right].copy()
                c,sn = np.cos(phase)[:,None],np.sin(phase)[:,None]
                bc[...,left],bc[...,right] = u*c-v*sn,u*sn+v*c
                rotated.append(bc)
            qr,kr = rotated
            v = vr[:,h*P:(h+1)*P,None].transpose(0,2,1)*px[h]
            v0 = initial['v'][b,h][None,:]*px[h]
            y,states = operand_forward(alpha[:,h],beta[:,h],gamma[:,h],qr,kr,v,
                initial['ssm'][b,h],initial['k'][b,h],v0,method,chunk)
            y += D[h]*v
            if g.norm:
                y = y/np.sqrt(np.mean(y*y,axis=-1,keepdims=True)+g.eps)*np.asarray(norm).reshape(H,P)[h]
            gate = z[:,h*P:(h+1)*P,None].transpose(0,2,1)*pz[h]
            output[b,:L,h*P:(h+1)*P] = np.sum(y*gate*sigmoid(gate)*po[h],axis=1)
            final['phase'][b,h] = phase[-1]
            final['ssm'][b,h] = states[-1]
            final['k'][b,h] = kr[-1]
            final['v'][b,h] = vr[-1,h*P:(h+1)*P]
    return (output if projected else output @ ow.T),final
