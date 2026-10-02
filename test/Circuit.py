from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
AREA = ROOT / "research" / "mamba3h"
PROVIDER = AREA / "integration" / "p1" / "provider" / "nsos_ext.cp312-win_amd64.pyd"
PROVIDER_SHA256 = "050f71f279bdd0df7b2f0fd00824376334cd8520db8343419df953990449232f"

SEEDS = (101, 211, 307, 401, 503)
UPDATES = 200
BATCH = 8
LR = 0.01
CLIP = 1.0
DIM = 8
MEMORY_CAPACITY = 64
TOP_K = 2

COMPONENT_TYPES = ("ALU", "TENSOR", "SRAM", "L2", "ROUTER", "PHY", "VRM", "PAD")
KINDS = (
    "DEFINE", "PLACE", "DRIVE", "SINK", "MOVE", "REWIRE", "NOOP",
    "Q_NET_TYPE", "Q_NET_ZONE", "Q_COMP_ZONE",
)
KIND_ID = {name: i for i, name in enumerate(KINDS)}
QUERY_KINDS = {"Q_NET_TYPE", "Q_NET_ZONE", "Q_COMP_ZONE"}
WRITE_KINDS = {"PLACE", "MOVE", "DRIVE", "REWIRE"}


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def derived_seed(seed: int, split: str, sample: int) -> int:
    raw = f"Circuit-v1|{seed}|{split}|{sample}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def event(kind: str, *, component=-1, net=-1, ctype=-1, zone=-1):
    return {
        "kind": kind,
        "component": int(component),
        "net": int(net),
        "ctype": int(ctype),
        "zone": int(zone),
    }


def entity_id(ev: dict) -> int:
    if ev["kind"] in {"DRIVE", "REWIRE", "Q_NET_TYPE", "Q_NET_ZONE"}:
        return ev["net"]
    if ev["kind"] in {"PLACE", "MOVE", "Q_COMP_ZONE"}:
        return 32 + ev["component"]
    return -1


def encode_event(ev: dict) -> np.ndarray:
    kind = KIND_ID[ev["kind"]]
    ent = entity_id(ev)
    return np.asarray(
        [
            kind / max(1, len(KINDS) - 1),
            0.0 if ent < 0 else ent / 63.0,
            0.0 if ev["component"] < 0 else (ev["component"] + 1) / 32.0,
            0.0 if ev["net"] < 0 else (ev["net"] + 1) / 32.0,
            0.0 if ev["ctype"] < 0 else (ev["ctype"] + 1) / 8.0,
            0.0 if ev["zone"] < 0 else (ev["zone"] + 1) / 8.0,
            float(ev["kind"] in QUERY_KINDS),
            float(ev["kind"] in WRITE_KINDS),
        ],
        dtype=np.float32,
    )


def generate_circuit(seed: int, split: str, sample: int, *, ood=False) -> dict:
    rng = np.random.default_rng(derived_seed(seed, split, sample))
    n_components = 16 if ood else 8
    n_nets = 16 if ood else 8
    length = 96 if ood else 48
    query_count = 16 if ood else 8
    mutation_count = 4 if ood else 2

    types = rng.integers(0, 8, size=n_components).tolist()
    zones = rng.integers(0, 8, size=n_components).tolist()
    drivers = rng.integers(0, n_components, size=n_nets).tolist()
    sinks = []
    for n in range(n_nets):
        choices = [c for c in range(n_components) if c != drivers[n]]
        sinks.append(int(rng.choice(choices)))

    events = []
    for c in rng.permutation(n_components):
        events.append(event("DEFINE", component=int(c), ctype=types[c]))
    for c in rng.permutation(n_components):
        events.append(event("PLACE", component=int(c), ctype=types[c], zone=zones[c]))
    for n in rng.permutation(n_nets):
        c = drivers[n]
        events.append(event("DRIVE", component=c, net=int(n), ctype=types[c], zone=zones[c]))
    for n in rng.permutation(n_nets):
        c = sinks[n]
        events.append(event("SINK", component=c, net=int(n), ctype=types[c], zone=zones[c]))

    # Long-range updates: move components and rewire nets before queries.
    for _ in range(mutation_count):
        c = int(rng.integers(0, n_components))
        zones[c] = int(rng.integers(0, 8))
        events.append(event("MOVE", component=c, ctype=types[c], zone=zones[c]))
    for _ in range(mutation_count):
        n = int(rng.integers(0, n_nets))
        c = int(rng.integers(0, n_components))
        drivers[n] = c
        events.append(event("REWIRE", component=c, net=n, ctype=types[c], zone=zones[c]))

    while len(events) + query_count < length:
        events.append(event("NOOP"))

    targets = [-100] * len(events)
    qtypes = ("Q_NET_TYPE", "Q_NET_ZONE", "Q_COMP_ZONE")
    for q in range(query_count):
        kind = qtypes[q % len(qtypes)]
        if kind == "Q_COMP_ZONE":
            c = int(rng.integers(0, n_components))
            events.append(event(kind, component=c))
            targets.append(zones[c])
        else:
            n = int(rng.integers(0, n_nets))
            c = drivers[n]
            events.append(event(kind, net=n))
            targets.append(types[c] if kind == "Q_NET_TYPE" else zones[c])

    assert len(events) == length and len(targets) == length
    for t, ev in enumerate(events):
        if ev["kind"] in QUERY_KINDS:
            assert 0 <= targets[t] < 8
        else:
            assert targets[t] == -100

    encoded = np.stack([encode_event(ev) for ev in events])
    fingerprint = sha256_bytes(encoded.tobytes())
    return {
        "events": events,
        "targets": targets,
        "encoded": encoded,
        "fingerprint": fingerprint,
        "components": [
            {"id": c, "type": types[c], "zone": zones[c]} for c in range(n_components)
        ],
        "nets": [
            {"id": n, "driver": drivers[n], "sink": sinks[n]} for n in range(n_nets)
        ],
    }


def build_split(seed: int, split: str, count: int, *, ood=False):
    rows = [generate_circuit(seed, split, i, ood=ood) for i in range(count)]
    fps = [r["fingerprint"] for r in rows]
    assert len(fps) == len(set(fps)), f"duplicate circuits inside {split}"
    x = np.stack([r["encoded"] for r in rows]).astype(np.float32)
    y = np.asarray([r["targets"] for r in rows], dtype=np.int64)
    return rows, x, y


def native_worker(seed: int) -> int:
    if sha256_bytes(PROVIDER.read_bytes()) != PROVIDER_SHA256:
        raise RuntimeError("frozen Mamba3 provider SHA drift")
    parent = str(PROVIDER.parent)
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(parent)
    sys.path.insert(0, parent)
    import nsos_ext as ns  # type: ignore

    raw = base64.b64decode(sys.stdin.buffer.read())
    data = np.load(io.BytesIO(raw))
    cfg = ns.Mamba3Config()
    cfg.expand = 1
    cfg.head_dim = 4
    cfg.state_dim = 128
    cfg.mimo = True
    cfg.mimo_rank = 1
    cfg.n_groups = 1
    cfg.seed = seed
    layer = ns.Mamba3Layer(DIM, cfg)
    layer.to(ns.Device.CPU)

    out = {}
    for name in data.files:
        arr = np.ascontiguousarray(data[name], dtype=np.float32)
        tape = layer.forward_owned(ns.Tensor.from_numpy(arr), ns.Mamba3State(), [])
        assert tape.audit_status() == [0] * len(arr)
        out[name] = tape.output().numpy().copy()
        tape.cancel()

    buf = io.BytesIO()
    np.savez_compressed(buf, **out)
    sys.stdout.write(base64.b64encode(buf.getvalue()).decode("ascii"))
    return 0


def frozen_native_features(arrays: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    p = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--native-worker", str(seed)],
        input=base64.b64encode(buf.getvalue()),
        capture_output=True,
        timeout=90,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="replace"))
    raw = base64.b64decode(p.stdout)
    data = np.load(io.BytesIO(raw))
    return {k: data[k].copy() for k in data.files}


def floorplan_ascii(circuit: dict) -> str:
    zones = [[] for _ in range(8)]
    for c in circuit["components"]:
        zones[c["zone"]].append(f"{COMPONENT_TYPES[c['type']]}{c['id']}")
    lines = ["GPU/BOARD FLOORPLAN (8 zonas)"]
    for row in range(2):
        cells = []
        for col in range(4):
            z = row * 4 + col
            label = ",".join(zones[z][:3]) or "empty"
            cells.append(f"Z{z}:{label[:22]:22}")
        lines.append(" | ".join(cells))
    lines.append("NETLIST (amostra)")
    for net in circuit["nets"][:12]:
        d, s = net["driver"], net["sink"]
        lines.append(
            f"N{net['id']:02d}: {COMPONENT_TYPES[circuit['components'][d]['type']]}{d}"
            f" -> {COMPONENT_TYPES[circuit['components'][s]['type']]}{s}"
        )
    return "\n".join(lines)


def run_benchmark(updates: int = UPDATES):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"

    import torch
    from torch import nn
    from research.mamba3h.algebra.operators import StructuredTransition
    from research.mamba3h.memory.slots import CausalSlotMemory, Control

    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    class CircuitModel(nn.Module):
        def __init__(self, arm: str, seed: int):
            super().__init__()
            self.arm = arm
            self.algebra = None
            self.memory = None
            if arm == "Mamba3H-C":
                self.algebra = StructuredTransition(
                    dim=DIM, rank=1, operations=10, commuting=False,
                    seed=seed, dtype=torch.float32,
                )
                torch.manual_seed(seed + 1000)
                self.memory = CausalSlotMemory(DIM, MEMORY_CAPACITY, TOP_K)
            torch.manual_seed(seed + 2000)
            self.head = nn.Linear(DIM, 8)

        def forward(self, x, events):
            batch, length, _ = x.shape
            ast = None
            memory_state = self.memory.initial_state(batch) if self.memory else None
            outputs = []
            for t in range(length):
                z = x[:, t]
                y = z
                if self.memory is not None:
                    previous = ast if ast is not None else torch.zeros_like(z)
                    op_ids = torch.tensor(
                        [KIND_ID[e[t]["kind"]] for e in events], dtype=torch.long
                    )
                    _, ast, _ = self.algebra.step(
                        z, previous, control={"op_id": op_ids}
                    )
                    ev = [e[t] for e in events]
                    write = torch.tensor([e["kind"] in WRITE_KINDS for e in ev], dtype=torch.bool)
                    read = torch.tensor([e["kind"] in QUERY_KINDS for e in ev], dtype=torch.bool)
                    entity = torch.tensor([entity_id(e) for e in ev], dtype=torch.long)
                    y, memory_state, _ = self.memory.step(
                        ast, memory_state, Control(write, read, entity)
                    )
                outputs.append(self.head(y))
            return torch.stack(outputs, dim=1)

    results = []
    example = None
    for seed in SEEDS:
        train_rows, train_raw, train_y = build_split(seed, "train", 64)
        val_rows, val_raw, val_y = build_split(seed, "validation", 32)
        test_rows, test_raw, test_y = build_split(seed, "test", 64)
        ood_rows, ood_raw, ood_y = build_split(seed, "ood", 64, ood=True)
        if example is None:
            example = ood_rows[0]

        all_fps = {
            "train": {r["fingerprint"] for r in train_rows},
            "validation": {r["fingerprint"] for r in val_rows},
            "test": {r["fingerprint"] for r in test_rows},
            "ood": {r["fingerprint"] for r in ood_rows},
        }
        names = list(all_fps)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                assert not all_fps[a] & all_fps[b], f"split leakage {a}/{b}"

        native = frozen_native_features(
            {
                "train": train_raw,
                "validation": val_raw,
                "test": test_raw,
                "ood": ood_raw,
            },
            seed,
        )
        raw_rms = np.sqrt(np.mean(train_raw ** 2, axis=(0, 1), dtype=np.float64))
        nat_rms = np.sqrt(np.mean(native["train"] ** 2, axis=(0, 1), dtype=np.float64))
        raw_scale = np.where(raw_rms > 0, raw_rms, 1.0)
        nat_scale = np.where(nat_rms > 0, nat_rms, 1.0)

        raw_map = {
            "train": train_raw, "validation": val_raw,
            "test": test_raw, "ood": ood_raw,
        }
        y_map = {
            "train": train_y, "validation": val_y,
            "test": test_y, "ood": ood_y,
        }
        rows_map = {
            "train": train_rows, "validation": val_rows,
            "test": test_rows, "ood": ood_rows,
        }
        x = {
            k: torch.tensor(
                (raw_map[k] / raw_scale + native[k] / nat_scale).astype(np.float32)
            )
            for k in raw_map
        }
        y = {k: torch.tensor(v, dtype=torch.long) for k, v in y_map.items()}
        ev = {k: [r["events"] for r in rows_map[k]] for k in rows_map}

        for arm in ("Mamba3-C", "Mamba3H-C"):
            model = CircuitModel(arm, seed)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            rng = np.random.default_rng(seed + 3000)
            losses = []
            started = time.perf_counter()
            model.train()
            for step in range(updates):
                ids = rng.choice(len(x["train"]), BATCH, replace=False)
                ix = torch.tensor(ids)
                opt.zero_grad(set_to_none=True)
                logits = model(x["train"][ix], [ev["train"][int(i)] for i in ids])
                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), y["train"][ix].flatten(), ignore_index=-100
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite loss seed={seed} arm={arm} step={step}")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                if not torch.isfinite(grad):
                    raise RuntimeError(f"nonfinite grad seed={seed} arm={arm} step={step}")
                opt.step()
                losses.append(float(loss.detach()))

            model.eval()
            metrics = {}
            with torch.no_grad():
                for split in ("validation", "test", "ood"):
                    correct = total = 0
                    type_counts = {}
                    for i in range(0, len(x[split]), 8):
                        logits = model(x[split][i:i + 8], ev[split][i:i + 8])
                        target = y[split][i:i + 8]
                        pred = logits.argmax(-1)
                        mask = target != -100
                        correct += int(((pred == target) & mask).sum())
                        total += int(mask.sum())
                        for b in range(len(ev[split][i:i + 8])):
                            for t, item in enumerate(ev[split][i + b]):
                                if item["kind"] in QUERY_KINDS:
                                    name = item["kind"]
                                    hit = int(pred[b, t] == target[b, t])
                                    c, n = type_counts.get(name, (0, 0))
                                    type_counts[name] = (c + hit, n + 1)
                    metrics[split] = {
                        "accuracy": correct / total,
                        "correct": correct,
                        "total": total,
                        "by_query": {
                            k: {"accuracy": c / n, "correct": c, "total": n}
                            for k, (c, n) in sorted(type_counts.items())
                        },
                    }

            row = {
                "seed": seed,
                "arm": arm,
                "updates": updates,
                "loss_first": losses[0],
                "loss_last": losses[-1],
                "loss_last25": float(np.mean(losses[-25:])),
                "parameters": sum(p.numel() for p in model.parameters()),
                "train_seconds": time.perf_counter() - started,
                "metrics": metrics,
            }
            results.append(row)
            print(
                f"[Circuit] seed={seed} arm={arm} "
                f"test={metrics['test']['accuracy']*100:.2f}% "
                f"ood={metrics['ood']['accuracy']*100:.2f}% "
                f"loss={losses[0]:.4f}->{losses[-1]:.4f}",
                flush=True,
            )

    table = {(r["seed"], r["arm"]): r for r in results}
    summary = {}
    for split in ("test", "ood"):
        base = np.asarray([
            table[(s, "Mamba3-C")]["metrics"][split]["accuracy"] for s in SEEDS
        ])
        hybrid = np.asarray([
            table[(s, "Mamba3H-C")]["metrics"][split]["accuracy"] for s in SEEDS
        ])
        delta = hybrid - base
        mean = float(delta.mean())
        half = float(2.776445105 * delta.std(ddof=1) / math.sqrt(len(delta)))
        summary[split] = {
            "mamba3_mean": float(base.mean()),
            "mamba3h_mean": float(hybrid.mean()),
            "paired_delta_mean": mean,
            "paired_t95": [mean - half, mean + half],
            "confirmatory_positive": bool(mean - half > 0.0),
        }

    report = {
        "benchmark": "Circuit-v1-native-Mamba3-vs-Mamba3H",
        "predeclared": {
            "seeds": list(SEEDS),
            "updates": updates,
            "batch": BATCH,
            "learning_rate": LR,
            "gradient_clip": CLIP,
            "train_examples": 64,
            "validation_examples": 32,
            "test_examples": 64,
            "ood_examples": 64,
            "normal_length": 48,
            "ood_length": 96,
            "criterion": "lower bound of paired 95% t interval > 0",
        },
        "provider_sha256": PROVIDER_SHA256,
        "scope": (
            "Mamba3-C = frozen native Mamba3 features + readout; "
            "Mamba3H-C = same backbone + StructuredTransition + entity-addressed "
            "CausalSlotMemory + readout. Synthetic hardware/netlist mapping benchmark."
        ),
        "results": results,
        "summary": summary,
    }

    print("\n" + "=" * 96)
    print("CIRCUIT BENCHMARK — MAMBA3-C vs MAMBA3H-C")
    print("=" * 96)
    for split in ("test", "ood"):
        s = summary[split]
        print(
            f"{split.upper():4s}: Mamba3-C={s['mamba3_mean']*100:.2f}%  "
            f"Mamba3H-C={s['mamba3h_mean']*100:.2f}%  "
            f"delta={s['paired_delta_mean']*100:+.2f} pp  "
            f"t95=[{s['paired_t95'][0]*100:+.2f}, {s['paired_t95'][1]*100:+.2f}] pp  "
            f"confirm={s['confirmatory_positive']}"
        )
    print("=" * 96)
    print(floorplan_ascii(example))
    print("=" * 96)
    print("CIRCUIT_JSON=" + json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Single-file circuit/GPU-board benchmark for native Mamba3 vs Mamba3H."
    )
    parser.add_argument("--native-worker", type=int)
    parser.add_argument("--updates", type=int, default=UPDATES)
    args = parser.parse_args()
    if args.native_worker is not None:
        return native_worker(args.native_worker)
    if args.updates <= 0:
        raise SystemExit("--updates must be positive")
    return run_benchmark(args.updates)


if __name__ == "__main__":
    raise SystemExit(main())
