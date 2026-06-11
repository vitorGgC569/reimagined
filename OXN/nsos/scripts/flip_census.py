"""
flip_census.py — EXP-FLIP-0: censo de flips ternarios entre checkpoints.

Roda em QUALQUER python com numpy (Colab CPU runtime, sem build): parser
puro-python do formato .bin do ModelSerializer. Espelha exatamente a regra de
quantizacao do BitLinear (bitlinear.cpp): t = clamp(round(W/scale), -1, +1),
scale = ||W||/sqrt(numel) (RMS por tensor).

Protocolo do arbitro (rodada 3): taxa de flips por tensor por intervalo;
fracao de REVERSOES A->B->A (com 3 ckpts) = oscilacao vs consolidacao;
tamanho entropy-coded do delta INCLUINDO tensores FP; baseline honesto.

Uso: python flip_census.py ckpt_A.bin ckpt_B.bin [ckpt_C.bin]
"""
from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

import numpy as np

MAGIC_TRAILER = 0x4E534E32


def read_ckpt(path):
    """-> dict {stable_name: (shape, np.float32 array)} (ordem preservada)."""
    data = Path(path).read_bytes()
    off = 0

    def u32():
        nonlocal off
        v = struct.unpack_from("<I", data, off)[0]
        off += 4
        return v

    magic, version, count = u32(), u32(), u32()
    out = {}
    for _ in range(count):
        nlen = u32()
        name = data[off:off + nlen].decode("utf-8", "replace"); off += nlen
        rank = u32()
        dims = [struct.unpack_from("<i", data, off + 4 * j)[0] for j in range(rank)]
        off += 4 * rank
        nbytes = u32()
        arr = np.frombuffer(data, dtype="<f4", count=nbytes // 4, offset=off).copy()
        off += nbytes
        out[name] = (dims, arr)
    return out


def ternary(w: np.ndarray) -> np.ndarray:
    scale = float(np.linalg.norm(w)) / max(np.sqrt(w.size), 1e-12)
    if scale <= 0:
        return np.zeros_like(w, dtype=np.int8)
    return np.clip(np.rint(w / scale), -1, 1).astype(np.int8)


def census(a, b, c=None):
    rows = []
    tot_w = tot_flips = tot_rev = 0
    fp_delta_bytes = 0
    flip_payload = bytearray()
    for name, (dims, wa) in a.items():
        if name not in b:
            continue
        wb = b[name][1]
        if wa.size != wb.size:
            continue
        is_mat = len(dims) == 2 and min(dims) >= 8
        is_embed = is_mat and max(dims) > 4000  # heuristica: embedding/head
        if is_mat and not is_embed:
            qa, qb = ternary(wa), ternary(wb)
            flips = int(np.count_nonzero(qa != qb))
            tot_w += wa.size
            tot_flips += flips
            rev = 0
            if c is not None and name in c:
                qc = ternary(c[name][1])
                rev = int(np.count_nonzero((qa != qb) & (qa == qc)))
                tot_rev += rev
            if flips:
                idx = np.flatnonzero(qa != qb).astype(np.uint32)
                flip_payload += idx.tobytes() + qb[qa != qb].tobytes()
            rows.append((name, wa.size, flips, rev))
        else:
            # tensores FP (embedding, normas, router, scales, vies): delta real
            d = (wb - wa).astype(np.float32)
            fp_delta_bytes += len(zlib.compress(d.tobytes(), 6))
    rows.sort(key=lambda r: -r[2])
    print(f"{'tensor':<44}{'pesos':>10}{'flips':>9}{'%':>8}{'rev':>7}")
    for name, n, f, rev in rows[:20]:
        print(f"{name:<44}{n:>10}{f:>9}{100.0*f/max(n,1):>7.3f}%{rev:>7}")
    flip_zip = len(zlib.compress(bytes(flip_payload), 6))
    print("-" * 78)
    print(f"[censo] pesos ternarios={tot_w:,}  flips={tot_flips:,} "
          f"({100.0*tot_flips/max(tot_w,1):.4f}%/intervalo)")
    if c is not None:
        print(f"[censo] reversoes A->B->A = {tot_rev:,} "
              f"({100.0*tot_rev/max(tot_flips,1):.1f}% dos flips = OSCILACAO; "
              f"resto = consolidacao)")
    print(f"[censo] delta-de-flips comprimido = {flip_zip/1e6:.3f} MB | "
          f"delta FP (embed/normas/router/scales) comprimido = {fp_delta_bytes/1e6:.3f} MB")
    print(f"[censo] TOTAL update funcional ~ {(flip_zip+fp_delta_bytes)/1e6:.3f} MB "
          f"(baseline: imagem packed ~{tot_w/4/1e6:.1f} MB + FP)")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    a = read_ckpt(sys.argv[1])
    b = read_ckpt(sys.argv[2])
    c = read_ckpt(sys.argv[3]) if len(sys.argv) > 3 else None
    print(f"[censo] A={sys.argv[1]}  B={sys.argv[2]}" + (f"  C={sys.argv[3]}" if c else ""))
    census(a, b, c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
