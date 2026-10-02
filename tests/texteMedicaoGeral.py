from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# This file is intentionally self-contained. It reads the NSOS runtime and the
# project's verified PT-BR task generator, but never writes outside this file.
REPO_ROOT = Path(__file__).resolve().parent.parent
NSOS_ROOT = REPO_ROOT / "OXN" / "nsos"
GPU_BUILD = NSOS_ROOT / "build-gm-hip-gpuopt"
SCRIPTS_DIR = NSOS_ROOT / "scripts"

SEED = 20261002
MIN_GPU_STEPS = 15000
os.environ["NSOS_MAMBA3_GPU_PROVIDER"] = "flash_fp32_replay_lds_v2"
os.environ["NSOS_MAMBA3_PROJECTION_PROVIDER"] = "exact_fp32"


class CharTokenizer:
    PAD = "<PAD>"
    BOS = "<BOS>"
    EOS = "<EOS>"

    def __init__(self, texts: Iterable[str]):
        chars = sorted({ch for text in texts for ch in text})
        self.tokens = [self.PAD, self.BOS, self.EOS, *chars]
        self.stoi = {token: i for i, token in enumerate(self.tokens)}
        self.itos = {i: token for token, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def bos_id(self) -> int:
        return self.stoi[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[self.EOS]

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        ids = [self.bos_id] if bos else []
        for ch in text:
            if ch not in self.stoi:
                raise RuntimeError(f"caractere fora do vocabulário de treino: {ch!r}")
            ids.append(self.stoi[ch])
        if eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        out: list[str] = []
        for token_id in ids:
            token = self.itos.get(int(token_id), "")
            if token == self.EOS:
                break
            if token not in (self.PAD, self.BOS):
                out.append(token)
        return "".join(out)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def resolve_nsos():
    candidates = sorted(GPU_BUILD.rglob("nsos_ext*.pyd"))
    if not candidates:
        raise RuntimeError(f"nsos_ext GPU não encontrado em {GPU_BUILD}")
    extension = candidates[0].resolve()
    sys.path.insert(0, str(extension.parent))
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(extension.parent))
    import nsos_ext as nsos  # type: ignore

    loaded = Path(nsos.__file__).resolve()
    if loaded != extension:
        raise RuntimeError(f"extensão errada carregada: {loaded} != {extension}")
    if not nsos.fast_gpu_supported():
        raise RuntimeError("runtime HIP carregado, mas fast_gpu_supported() retornou false")
    if hasattr(nsos, "set_strict_gpu_execution"):
        nsos.set_strict_gpu_execution(True)
    return nsos, extension


def load_verified_tasks():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import ptbr_verified_tasks as verified  # type: ignore
    return verified


def make_exact_split(verified, train_per_family: int, eval_per_family: int):
    train = {family: [] for family in verified.FAMILIES}
    held = {family: [] for family in verified.FAMILIES}
    index = 0
    while (
        any(len(rows) < train_per_family for rows in train.values())
        or any(len(rows) < eval_per_family for rows in held.values())
    ):
        record = verified.generate(SEED, index)
        proof = verified.verify(record["prompt"], record["answer"])
        family = proof["family"]
        target = held if proof["split"] == "eval" else train
        limit = eval_per_family if proof["split"] == "eval" else train_per_family
        if len(target[family]) < limit:
            target[family].append(
                {
                    "prompt": record["prompt"],
                    "answer": record["answer"],
                    "family": family,
                    "problem_id": proof["problem_id"],
                }
            )
        index += 1
        if index > 100_000:
            raise RuntimeError("não foi possível preencher o split verificado")
    train_rows = [row for family in verified.FAMILIES for row in train[family]]
    eval_rows = [row for family in verified.FAMILIES for row in held[family]]
    if {row["problem_id"] for row in train_rows} & {row["problem_id"] for row in eval_rows}:
        raise RuntimeError("vazamento de problema entre train/eval")
    return train_rows, eval_rows


def make_lm_split(train_count: int, eval_count: int):
    subjects = [
        "O sistema", "A equipe", "O modelo", "O laboratório", "A pesquisadora",
        "O servidor", "A avaliação", "O experimento",
    ]
    verbs = ["mede", "analisa", "registra", "compara", "valida", "observa"]
    objects = [
        "a latência", "a precisão", "a memória", "o contexto", "a resposta",
        "o treinamento", "a inferência", "a estabilidade",
    ]
    tails = [
        "em português brasileiro",
        "com dados mantidos fora do treino",
        "antes da próxima execução",
        "sem alterar a arquitetura",
        "com critérios reproduzíveis",
        "durante a validação final",
    ]
    rows = []
    for subject in subjects:
        for verb in verbs:
            for obj in objects:
                for tail in tails:
                    text = f"{subject} {verb} {obj} {tail}."
                    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    rows.append((digest, text))
    rows.sort()
    eval_rows = [text for digest, text in rows if int(digest[:8], 16) % 10 == 0][:eval_count]
    eval_set = set(eval_rows)
    train_rows = [text for _, text in rows if text not in eval_set][:train_count]
    if len(train_rows) < train_count or len(eval_rows) < eval_count:
        raise RuntimeError("corpus LM controlado insuficiente")
    if set(train_rows) & set(eval_rows):
        raise RuntimeError("vazamento textual no corpus LM")
    return train_rows, eval_rows


def exact_prompt(row: dict) -> str:
    return "Pergunta: " + row["prompt"] + "\nResposta:"


def build_model(nsos, vocab_size: int):
    cfg = nsos.ModelConfig()
    cfg.architecture_schema_version = 3
    cfg.mamba3_enabled = True
    cfg.mamba3_state_dim = 128
    cfg.num_layers = 2
    cfg.d_model = 128
    cfg.vocab_size = vocab_size
    cfg.n_heads = 4
    cfg.n_kv_heads = 2
    cfg.use_moe = False
    cfg.use_kan = False
    cfg.use_ttt = False
    cfg.dropout = 0.0
    cfg.max_context_tokens = 256
    cfg.attention_period = 1_000_000
    cfg.attention_slot = 999_999
    nsos.set_seed(SEED)
    if hasattr(nsos, "set_deterministic_reductions"):
        nsos.set_deterministic_reductions(True)
    model = nsos.JambaModel(cfg, nsos.Device.GPU)
    model.to(nsos.Device.GPU)
    return model, cfg


def log_probs(row: np.ndarray) -> np.ndarray:
    values = np.asarray(row, dtype=np.float64)
    maximum = float(np.max(values))
    shifted = values - maximum
    return shifted - math.log(float(np.exp(shifted).sum()))


def forward_logits(model, ids: Sequence[int], vocab_size: int) -> np.ndarray:
    model.reset_session()
    logits = np.asarray(model.forward_ids(list(ids), None).cpu().numpy(), dtype=np.float64)
    logits = logits.reshape(len(ids), vocab_size)
    if not np.isfinite(logits).all():
        raise RuntimeError("logits não finitos")
    return logits


def score_targets(model, prompt: Sequence[int], target: Sequence[int], vocab_size: int):
    full = list(prompt) + list(target)
    logits = forward_logits(model, full, vocab_size)
    correct = 0
    nll = 0.0
    for offset, token in enumerate(target):
        row = logits[len(prompt) - 1 + offset]
        pred = int(np.argmax(row))
        correct += int(pred == int(token))
        nll -= float(log_probs(row)[int(token)])
    return correct, len(target), nll


def greedy_answer(model, tokenizer: CharTokenizer, prompt_text: str, max_tokens: int) -> str:
    ids = tokenizer.encode(prompt_text, bos=True)
    generated: list[int] = []
    for _ in range(max_tokens):
        logits = forward_logits(model, ids, tokenizer.vocab_size)
        token = int(np.argmax(logits[-1]))
        if token == tokenizer.eos_id:
            break
        generated.append(token)
        ids.append(token)
    return tokenizer.decode(generated)


def train_campaign(
    nsos,
    model,
    tokenizer: CharTokenizer,
    exact_train: list[dict],
    lm_train: list[str],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
):
    exact_pairs = [
        (
            tokenizer.encode(exact_prompt(row), bos=True),
            tokenizer.encode(row["answer"], eos=True),
        )
        for row in exact_train
    ]
    lm_pairs = [
        (
            [tokenizer.bos_id],
            tokenizer.encode(text, eos=True),
        )
        for text in lm_train
    ]
    trainer = nsos.Trainer(model, learning_rate)
    trainer.weight_decay = 1e-4
    trainer.max_grad_norm = 1.0
    trainer.warmup_steps = max(5, min(50, steps // 10))
    trainer.total_training_steps = steps
    trainer.eos_token_id = tokenizer.eos_id
    trainer.first_token_loss_scale = 1.0
    trainer.eos_loss_scale = 1.0

    rng = random.Random(SEED + 1)
    model.set_training_mode(True)
    losses: list[float] = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        prompts: list[list[int]] = []
        answers: list[list[int]] = []
        for sample_idx in range(batch_size):
            use_exact = (sample_idx % 2 == 0)
            pool = exact_pairs if use_exact else lm_pairs
            prompt, answer = pool[rng.randrange(len(pool))]
            prompts.append(prompt)
            answers.append(answer)
        loss = float(trainer.train_supervised_batch(prompts, answers))
        if not math.isfinite(loss):
            raise RuntimeError(f"loss não finita no passo {step}: {loss}")
        losses.append(loss)
        if step in {1, max(1, steps // 4), max(1, steps // 2), max(1, 3 * steps // 4), steps}:
            tail = losses[-min(25, len(losses)) :]
            print(
                f"[treino] passo={step}/{steps} loss={loss:.6f} "
                f"media25={sum(tail)/len(tail):.6f}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    model.set_training_mode(False)
    return trainer, losses, elapsed


def evaluate(
    model,
    tokenizer: CharTokenizer,
    exact_eval: list[dict],
    lm_eval: list[str],
):
    teacher_correct = 0
    teacher_total = 0
    exact_correct = 0
    exact_by_family: dict[str, list[int]] = {}

    for row in exact_eval:
        prompt_text = exact_prompt(row)
        prompt = tokenizer.encode(prompt_text, bos=True)
        target = tokenizer.encode(row["answer"], eos=True)
        correct, total, _ = score_targets(model, prompt, target, tokenizer.vocab_size)
        teacher_correct += correct
        teacher_total += total

        generated = greedy_answer(
            model,
            tokenizer,
            prompt_text,
            max_tokens=max(8, len(row["answer"]) + 8),
        )
        hit = int(generated == row["answer"])
        exact_correct += hit
        exact_by_family.setdefault(row["family"], []).append(hit)

    lm_nll = 0.0
    lm_tokens = 0
    for text in lm_eval:
        prompt = [tokenizer.bos_id]
        target = tokenizer.encode(text, eos=True)
        _, total, nll = score_targets(model, prompt, target, tokenizer.vocab_size)
        lm_nll += nll
        lm_tokens += total

    ppl = math.exp(lm_nll / lm_tokens)
    if not math.isfinite(ppl) or ppl < 1.0 or lm_tokens <= 0:
        raise RuntimeError("perplexidade inválida")
    return {
        "exact_match_ptbr": exact_correct / len(exact_eval),
        "exact_match_count": exact_correct,
        "exact_match_total": len(exact_eval),
        "exact_match_by_family": {
            family: {
                "correct": sum(values),
                "total": len(values),
                "rate": sum(values) / len(values),
            }
            for family, values in sorted(exact_by_family.items())
        },
        "teacher_token_accuracy": teacher_correct / teacher_total,
        "teacher_token_correct": teacher_correct,
        "teacher_token_total": teacher_total,
        "perplexity_ptbr": ppl,
        "perplexity_nll_sum": lm_nll,
        "perplexity_scored_tokens": lm_tokens,
    }



def run_mamba3h_validation() -> int:
    """Read-only five-seed replay of frozen-backbone Mamba3H P0 (M0 vs corrected M3)."""
    area = REPO_ROOT / "research" / "mamba3h"
    protocol_path = area / "manifests" / "pilot-v2.json"
    prepared = area / "review" / "pilot-v2"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    # Import only the frozen research adapter definitions.  No files are written.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from research.mamba3h.integration import paired_pilot as pp  # type: ignore

    pp.PROTOCOL_FILE = protocol_path
    pp.PROTOCOL = protocol
    Model, torch = pp.model_classes()
    extended_updates = 200  # exploratory 10x schedule; frozen P0 remains 20

    def load_seed(seed: int):
        sd = prepared / f"seed-{seed}"
        episodes = json.loads((sd / "episodes.json").read_text(encoding="utf-8"))
        inputs = np.load(sd / "input.npz")
        native = np.load(sd / "native.npz")
        x = {k: torch.tensor(inputs[k] + native[k]) for k in inputs.files}
        targets = {
            k: torch.tensor(
                [[5 if value == 125 else value for value in ep["targets"]] for ep in eps]
            )
            for k, eps in episodes.items()
        }
        events = {k: [ep["events"] for ep in eps] for k, eps in episodes.items()}
        return sd, x, targets, events

    def fit_eval(seed: int, arm: str):
        sd, x, targets, events = load_seed(seed)
        model = Model(arm, seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(protocol["learning_rate"]))
        rng = np.random.default_rng(seed + 3000)
        losses = []
        model.train()
        for _ in range(extended_updates):
            ix_np = rng.choice(
                len(x["train"]), int(protocol["minibatch"]), replace=False
            )
            ix = torch.tensor(ix_np)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(
                x["train"][ix], [events["train"][int(i)] for i in ix_np]
            )
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1),
                targets["train"][ix].flatten(),
                ignore_index=-100,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Mamba3H loss não finita seed={seed} arm={arm}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(protocol["gradient_clip"])
            )
            if not torch.isfinite(grad):
                raise RuntimeError(f"Mamba3H gradiente não finito seed={seed} arm={arm}")
            optimizer.step()
            losses.append(float(loss.detach()))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(x["test"]), 8):
                logits, _ = model(x["test"][i:i + 8], events["test"][i:i + 8])
                y = targets["test"][i:i + 8]
                mask = y != -100
                correct += int(((logits.argmax(-1) == y) & mask).sum())
                total += int(mask.sum())
        accuracy = correct / total

        # Compare with the previously frozen receipt without trusting it as the result.
        receipt = json.loads((sd / f"{arm}.json").read_text(encoding="utf-8"))
        frozen_accuracy = float(receipt["evaluations"]["test"]["accuracy"])
        return {
            "seed": seed,
            "arm": arm,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "frozen_accuracy": frozen_accuracy,
            "matches_frozen_20_update_receipt": (
                extended_updates == int(protocol["updates"])
                and math.isclose(accuracy, frozen_accuracy, rel_tol=0.0, abs_tol=0.0)
            ),
        }

    rows = []
    for seed in protocol["seeds"]:
        for arm in ("M0", "M3"):
            row = fit_eval(int(seed), arm)
            rows.append(row)
            print(
                f"[Mamba3H] seed={seed} arm={arm} "
                f"test={row['accuracy']*100:.4f}% "
                f"loss={row['loss_first']:.4f}->{row['loss_last']:.4f} "
                f"frozen20={row['frozen_accuracy']*100:.4f}%",
                flush=True,
            )

    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["arm"]] = row
    deltas = [
        by_seed[int(seed)]["M3"]["accuracy"] - by_seed[int(seed)]["M0"]["accuracy"]
        for seed in protocol["seeds"]
    ]
    m0 = np.array([by_seed[int(seed)]["M0"]["accuracy"] for seed in protocol["seeds"]])
    m3 = np.array([by_seed[int(seed)]["M3"]["accuracy"] for seed in protocol["seeds"]])
    delta = np.array(deltas, dtype=np.float64)
    mean = float(delta.mean())
    half = float(2.776445105 * delta.std(ddof=1) / math.sqrt(len(delta)))
    all_match = None  # extended schedule intentionally differs from frozen 20-update P0

    report = {
        "label": protocol["label"],
        "protocol": protocol["protocol"],
        "seeds": list(protocol["seeds"]),
        "updates_per_arm": extended_updates,
        "frozen_protocol_updates": int(protocol["updates"]),
        "exploratory_extended_training": True,
        "M0_mean_accuracy": float(m0.mean()),
        "M3_mean_accuracy": float(m3.mean()),
        "paired_delta_mean": mean,
        "paired_delta_t95": [mean - half, mean + half],
        "all_replays_match_frozen_receipts": all_match,
        "native_end_to_end": False,
        "scope": (
            "actual native frozen Mamba-3 features + external corrected M3 "
            "(StructuredTransition -> CausalSlotMemory), read-only replay"
        ),
    }
    print("\n" + "=" * 80)
    print("MAMBA-3H READ-ONLY VALIDATION")
    print("=" * 80)
    print(f"M0 mean test accuracy: {report['M0_mean_accuracy']*100:.4f}%")
    print(f"M3 mean test accuracy: {report['M3_mean_accuracy']*100:.4f}%")
    print(f"M3-M0 paired delta:    {report['paired_delta_mean']*100:+.4f} pp")
    print(
        "95% t interval:         "
        f"[{report['paired_delta_t95'][0]*100:+.4f}, "
        f"{report['paired_delta_t95'][1]*100:+.4f}] pp"
    )
    print("Replay == receipts:     N/A — treino estendido 200 vs P0 congelado 20")
    print("Native end-to-end:      NÃO — frozen native backbone + external adapters")
    print("=" * 80)
    print("MAMBA3H_JSON=" + json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0



CONFIRMATORY_MAMBA3H_SEEDS = (101, 211, 307, 401, 503)
CONFIRMATORY_MAMBA3H_UPDATES = 200
CONFIRMATORY_PREVIOUS_DELTA = 0.07265625
FROZEN_MAMBA3_PROVIDER_SHA256 = "050f71f279bdd0df7b2f0fd00824376334cd8520db8343419df953990449232f"


def _mamba3h_native_child(seed: int) -> int:
    """Compute frozen native Mamba-3 features from NPZ bytes on stdin; write NPZ bytes to stdout."""
    # The main PT-BR benchmark opts into an optimized GPU provider globally.
    # The frozen Mamba3H backbone is a CPU/dense-reference identity; isolate it.
    os.environ["NSOS_MAMBA3_GPU_PROVIDER"] = "dense_reference"
    os.environ["NSOS_MAMBA3_PROJECTION_PROVIDER"] = "exact_fp32"
    provider_dir = (
        REPO_ROOT
        / "research"
        / "mamba3h"
        / "integration"
        / "p1"
        / "provider"
    )
    extension = provider_dir / "nsos_ext.cp312-win_amd64.pyd"
    if sha256_file(extension) != FROZEN_MAMBA3_PROVIDER_SHA256:
        raise RuntimeError("Mamba3H frozen private provider SHA drift")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(provider_dir))
    sys.path.insert(0, str(provider_dir))
    import nsos_ext as ns  # type: ignore

    raw = sys.stdin.buffer.read()
    src = np.load(io.BytesIO(raw))
    cfg = ns.Mamba3Config()
    cfg.expand = 1
    cfg.head_dim = 4
    cfg.state_dim = 128
    cfg.mimo = True
    cfg.mimo_rank = 1
    cfg.n_groups = 1
    cfg.seed = int(seed)
    layer = ns.Mamba3Layer(8, cfg)
    layer.to(ns.Device.CPU)

    result = {}
    for split in src.files:
        arr = np.ascontiguousarray(src[split], dtype=np.float32)
        tensor = ns.Tensor.from_numpy(arr)
        tape = layer.forward_owned(tensor, ns.Mamba3State(), [])
        audit = tape.audit_status()
        if audit != [0] * len(arr):
            raise RuntimeError(f"native audit failure seed={seed} split={split}: {audit}")
        result[split] = tape.output().numpy().copy()
        tape.cancel()

    out = io.BytesIO()
    np.savez(out, **result)
    sys.stdout.buffer.write(out.getvalue())
    return 0


def _native_features_in_memory(seed: int, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    payload = io.BytesIO()
    np.savez(payload, **arrays)
    cmd = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--mamba3h-native-child",
        "--native-seed",
        str(seed),
    ]
    proc = subprocess.run(
        cmd,
        input=payload.getvalue(),
        cwd=REPO_ROOT,
        capture_output=True,
        timeout=90,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"native feature subprocess failed seed={seed}: "
            + proc.stderr.decode("utf-8", errors="replace")
        )
    packed = np.load(io.BytesIO(proc.stdout))
    return {name: packed[name].copy() for name in packed.files}


def run_mamba3h_confirmation() -> int:
    """Fresh five-seed confirmatory M0-vs-M3 study; all data/features stay in memory."""
    area = REPO_ROOT / "research" / "mamba3h"
    pilot_protocol = json.loads(
        (area / "manifests" / "pilot-v2.json").read_text(encoding="utf-8")
    )

    confirmatory = {
        "study": "mamba3h-fresh-confirmatory-200-v1",
        "predeclared_before_training": True,
        "prior_seed_absence_checked": True,
        "seeds": list(CONFIRMATORY_MAMBA3H_SEEDS),
        "arms": ["M0", "M3"],
        "task": "inst",
        "group": "s5",
        "regime": "within_group",
        "difficulty": dict(pilot_protocol["difficulty"]),
        "samples": dict(pilot_protocol["samples"]),
        "updates": CONFIRMATORY_MAMBA3H_UPDATES,
        "minibatch": int(pilot_protocol["minibatch"]),
        "optimizer": "Adam",
        "learning_rate": float(pilot_protocol["learning_rate"]),
        "gradient_clip": float(pilot_protocol["gradient_clip"]),
        "memory_input_M3": "algebra_pre_read",
        "selection": "fixed schedule; no validation/test model selection; test evaluated once after training",
        "primary_endpoint": "paired test accuracy delta M3-M0 across five seeds",
        "analysis": "mean paired delta and conventional two-sided t95 interval df=4",
        "confirmatory_criterion": "lower bound of paired t95 interval > 0",
        "previous_exploratory_delta_reference": CONFIRMATORY_PREVIOUS_DELTA,
        "native_backbone": "actual frozen private CPU Mamba3 provider; no end-to-end native training",
        "provider_sha256": FROZEN_MAMBA3_PROVIDER_SHA256,
    }
    print(
        "PREDECLARED_CONFIRMATORY="
        + json.dumps(confirmatory, ensure_ascii=False, sort_keys=True),
        flush=True,
    )

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from research.mamba3h.benchmarks.generation import generate  # type: ignore
    from research.mamba3h.benchmarks.oracle import solve  # type: ignore
    from research.mamba3h.benchmarks.schema import (  # type: ignore
        encode_numeric,
        fingerprint,
        model_view,
    )

    prepared: dict[int, dict[str, object]] = {}
    global_inputs: set[str] = set()
    split_manifest = []

    # Data and native features are prepared fully before torch/adapters are imported.
    for seed in CONFIRMATORY_MAMBA3H_SEEDS:
        episodes: dict[str, list[dict]] = {}
        arrays: dict[str, np.ndarray] = {}
        fingerprints: dict[str, list[str]] = {}
        for split, count in confirmatory["samples"].items():
            eps = generate(
                "inst",
                int(seed),
                str(split),
                int(count),
                **confirmatory["difficulty"],
            )
            for ep in eps:
                if solve(ep) != ep["targets"]:
                    raise RuntimeError(
                        f"independent oracle mismatch seed={seed} split={split}"
                    )
            fps = [fingerprint(model_view(ep)) for ep in eps]
            if len(fps) != len(set(fps)):
                raise RuntimeError(f"duplicate examples seed={seed} split={split}")
            for fp in fps:
                if fp in global_inputs:
                    raise RuntimeError(f"cross-seed/split duplicate input: {fp}")
                global_inputs.add(fp)
            episodes[str(split)] = eps
            fingerprints[str(split)] = fps
            arrays[str(split)] = np.asarray(
                [encode_numeric(ep, 8) for ep in eps], dtype=np.float32
            )

        if (
            set(fingerprints["train"]) & set(fingerprints["validation"])
            or set(fingerprints["train"]) & set(fingerprints["test"])
            or set(fingerprints["validation"]) & set(fingerprints["test"])
        ):
            raise RuntimeError(f"split leakage seed={seed}")

        native = _native_features_in_memory(int(seed), arrays)
        prepared[int(seed)] = {
            "episodes": episodes,
            "arrays": arrays,
            "native": native,
        }
        split_manifest.append(
            {
                "seed": int(seed),
                "train_sha256": stable_hash(
                    [model_view(ep) for ep in episodes["train"]]
                ),
                "validation_sha256": stable_hash(
                    [model_view(ep) for ep in episodes["validation"]]
                ),
                "test_sha256": stable_hash(
                    [model_view(ep) for ep in episodes["test"]]
                ),
                "native_train_sha256": hashlib.sha256(
                    native["train"].tobytes()
                ).hexdigest(),
                "native_validation_sha256": hashlib.sha256(
                    native["validation"].tobytes()
                ).hexdigest(),
                "native_test_sha256": hashlib.sha256(
                    native["test"].tobytes()
                ).hexdigest(),
            }
        )

    print(
        "FRESH_SPLIT_MANIFEST="
        + json.dumps(split_manifest, ensure_ascii=False, sort_keys=True),
        flush=True,
    )

    # Import adapters only after native feature extraction is finished, preserving
    # the original separate-native-process OpenMP boundary.
    from research.mamba3h.integration import paired_pilot as pp  # type: ignore

    pp.PROTOCOL = dict(pilot_protocol)
    pp.PROTOCOL["memory_input_M3"] = "algebra_pre_read"
    Model, torch = pp.model_classes()

    def train_eval(seed: int, arm: str) -> dict:
        item = prepared[seed]
        episodes = item["episodes"]
        arrays = item["arrays"]
        native = item["native"]
        x = {
            split: torch.tensor(arrays[split] + native[split])
            for split in ("train", "validation", "test")
        }
        targets = {
            split: torch.tensor(
                [
                    [5 if value == 125 else value for value in ep["targets"]]
                    for ep in episodes[split]
                ]
            )
            for split in ("train", "validation", "test")
        }
        events = {
            split: [ep["events"] for ep in episodes[split]]
            for split in ("train", "validation", "test")
        }

        model = Model(arm, seed)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(confirmatory["learning_rate"])
        )
        rng = np.random.default_rng(seed + 3000)
        losses: list[float] = []
        model.train()
        for _ in range(int(confirmatory["updates"])):
            ids = rng.choice(
                len(x["train"]),
                int(confirmatory["minibatch"]),
                replace=False,
            )
            ix = torch.tensor(ids)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(
                x["train"][ix],
                [events["train"][int(i)] for i in ids],
            )
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1),
                targets["train"][ix].flatten(),
                ignore_index=-100,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"nonfinite loss seed={seed} arm={arm}"
                )
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(confirmatory["gradient_clip"])
            )
            if not torch.isfinite(grad):
                raise FloatingPointError(
                    f"nonfinite gradient seed={seed} arm={arm}"
                )
            optimizer.step()
            losses.append(float(loss.detach()))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(x["test"]), 8):
                logits, _ = model(
                    x["test"][i : i + 8],
                    events["test"][i : i + 8],
                )
                y = targets["test"][i : i + 8]
                mask = y != -100
                correct += int(((logits.argmax(-1) == y) & mask).sum())
                total += int(mask.sum())
        return {
            "seed": seed,
            "arm": arm,
            "accuracy": correct / total,
            "correct": correct,
            "total": total,
            "loss_first": losses[0],
            "loss_last": losses[-1],
        }

    # Fixed order, no adaptation after seeing any test result.
    rows = []
    for seed in CONFIRMATORY_MAMBA3H_SEEDS:
        pair = []
        for arm in ("M0", "M3"):
            pair.append(train_eval(int(seed), arm))
        rows.extend(pair)
        print(
            f"[confirmatory] seed={seed} "
            f"M0={pair[0]['accuracy']*100:.4f}% "
            f"M3={pair[1]['accuracy']*100:.4f}% "
            f"delta={(pair[1]['accuracy']-pair[0]['accuracy'])*100:+.4f}pp",
            flush=True,
        )

    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["arm"]] = row
    m0 = np.asarray(
        [by_seed[s]["M0"]["accuracy"] for s in CONFIRMATORY_MAMBA3H_SEEDS],
        dtype=np.float64,
    )
    m3 = np.asarray(
        [by_seed[s]["M3"]["accuracy"] for s in CONFIRMATORY_MAMBA3H_SEEDS],
        dtype=np.float64,
    )
    deltas = m3 - m0
    mean = float(deltas.mean())
    half = float(2.776445105 * deltas.std(ddof=1) / math.sqrt(len(deltas)))
    interval = [mean - half, mean + half]
    confirmed = interval[0] > 0.0

    report = {
        "study": confirmatory["study"],
        "confirmatory": True,
        "protocol": confirmatory,
        "split_manifest": split_manifest,
        "rows": rows,
        "M0_mean_accuracy": float(m0.mean()),
        "M3_mean_accuracy": float(m3.mean()),
        "paired_deltas": deltas.tolist(),
        "paired_delta_mean": mean,
        "paired_delta_t95": interval,
        "criterion_met": confirmed,
        "delta_vs_previous_exploratory": mean - CONFIRMATORY_PREVIOUS_DELTA,
        "all_five_seed_deltas_positive": bool(np.all(deltas > 0)),
        "native_end_to_end": False,
    }

    print("\n" + "=" * 80)
    print("MAMBA-3H FRESH CONFIRMATORY RESULT")
    print("=" * 80)
    print(f"M0 mean test accuracy: {report['M0_mean_accuracy']*100:.4f}%")
    print(f"M3 mean test accuracy: {report['M3_mean_accuracy']*100:.4f}%")
    print(f"M3-M0 paired delta:    {mean*100:+.4f} pp")
    print(f"95% t interval:         [{interval[0]*100:+.4f}, {interval[1]*100:+.4f}] pp")
    print(f"All 5 deltas positive: {report['all_five_seed_deltas_positive']}")
    print(f"Confirmatory criterion: {'PASS' if confirmed else 'FAIL'}")
    print(
        f"Previous exploratory:   {CONFIRMATORY_PREVIOUS_DELTA*100:+.4f} pp; "
        f"fresh difference={(mean-CONFIRMATORY_PREVIOUS_DELTA)*100:+.4f} pp"
    )
    print("Native end-to-end:      NÃO — frozen native backbone + external adapters")
    print("=" * 80)
    print(
        "MAMBA3H_CONFIRMATORY_JSON="
        + json.dumps(report, ensure_ascii=False, sort_keys=True)
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Medição geral PT-BR controlada do NSOS/Mamba-3 sem alterar a arquitetura."
    )
    parser.add_argument("--steps", type=int, default=MIN_GPU_STEPS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--exact-train-per-family", type=int, default=50)
    parser.add_argument("--exact-eval-per-family", type=int, default=5)
    parser.add_argument("--lm-train", type=int, default=240)
    parser.add_argument("--lm-eval", type=int, default=48)
    parser.add_argument("--mamba3h-validate", action="store_true")
    parser.add_argument("--mamba3h-confirm", action="store_true")
    parser.add_argument("--mamba3h-native-child", action="store_true")
    parser.add_argument("--native-seed", type=int, default=11)
    args = parser.parse_args()
    if args.mamba3h_native_child:
        return _mamba3h_native_child(args.native_seed)
    if args.mamba3h_confirm:
        return run_mamba3h_confirmation()
    if args.mamba3h_validate:
        return run_mamba3h_validation()
    if args.steps < MIN_GPU_STEPS or args.batch_size <= 0:
        raise SystemExit(f"esta campanha GPU exige pelo menos {MIN_GPU_STEPS} passos")

    nsos, extension = resolve_nsos()
    verified = load_verified_tasks()
    exact_train, exact_eval = make_exact_split(
        verified, args.exact_train_per_family, args.exact_eval_per_family
    )
    lm_train, lm_eval = make_lm_split(args.lm_train, args.lm_eval)

    tokenizer_texts = [
        *(exact_prompt(row) + row["answer"] for row in exact_train),
        *lm_train,
    ]
    tokenizer = CharTokenizer(tokenizer_texts)

    # The eval set may contain new combinations, but not new characters.
    for row in exact_eval:
        tokenizer.encode(exact_prompt(row) + row["answer"])
        verified.verify(row["prompt"], row["answer"])
    for text in lm_eval:
        tokenizer.encode(text)

    model, cfg = build_model(nsos, tokenizer.vocab_size)
    param_count = int(sum(int(p.data.size) for p in model.parameters()))

    identity = {
        "seed": SEED,
        "runtime": str(extension),
        "runtime_sha256": sha256_file(extension),
        "verified_task_version": str(verified.VERSION),
        "exact_train_sha256": stable_hash(exact_train),
        "exact_eval_sha256": stable_hash(exact_eval),
        "lm_train_sha256": stable_hash(lm_train),
        "lm_eval_sha256": stable_hash(lm_eval),
        "vocab_size": tokenizer.vocab_size,
        "parameters": param_count,
        "architecture_schema_version": int(cfg.architecture_schema_version),
        "mamba3_enabled": bool(cfg.mamba3_enabled),
        "mamba3_state_dim": int(cfg.mamba3_state_dim),
        "num_layers": int(cfg.num_layers),
        "d_model": int(cfg.d_model),
        "device": "GPU",
        "gpu_backend": "hip",
        "mamba3_gpu_provider": os.environ["NSOS_MAMBA3_GPU_PROVIDER"],
        "mamba3_projection_provider": os.environ["NSOS_MAMBA3_PROJECTION_PROVIDER"],
    }
    print("[identidade] " + json.dumps(identity, ensure_ascii=False, sort_keys=True), flush=True)

    trainer, losses, train_seconds = train_campaign(
        nsos,
        model,
        tokenizer,
        exact_train,
        lm_train,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    metrics = evaluate(model, tokenizer, exact_eval, lm_eval)

    report = {
        "benchmark": "texteMedicaoGeral-controlled-ptbr-gpu-5x-v2",
        "scope": (
            "Campanha controlada PT-BR em memória. Exact match usa as cinco famílias "
            "do gerador/verificador independente ptbr_verified_tasks. Perplexidade usa "
            "frases PT-BR held-out do próprio teste. Não substitui Wikipedia-PT nem o "
            "verified-pilot de 12M tokens."
        ),
        "identity": identity,
        "campaign": {
            "planned_steps": args.steps,
            "completed_steps": int(trainer.global_step_count),
            "recipe_completed": int(trainer.global_step_count) == args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "exact_train_examples": len(exact_train),
            "exact_eval_examples": len(exact_eval),
            "lm_train_examples": len(lm_train),
            "lm_eval_examples": len(lm_eval),
            "train_seconds": train_seconds,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "loss_last25_mean": sum(losses[-min(25, len(losses)):]) / min(25, len(losses)),
            "tokens_processed": int(trainer.tokens_processed),
            "tokens_committed": int(trainer.tokens_committed),
        },
        "metrics": metrics,
    }
    if not report["campaign"]["recipe_completed"]:
        raise RuntimeError(
            f"campanha incompleta: {trainer.global_step_count}/{args.steps} passos"
        )

    print("\n" + "=" * 80)
    print("MEDIÇÃO GERAL PT-BR — RESULTADO")
    print("=" * 80)
    print(
        f"Exact Match PT-BR:       {metrics['exact_match_ptbr'] * 100:.2f}% "
        f"({metrics['exact_match_count']}/{metrics['exact_match_total']})"
    )
    print(
        f"Teacher-token accuracy: {metrics['teacher_token_accuracy'] * 100:.2f}% "
        f"({metrics['teacher_token_correct']}/{metrics['teacher_token_total']})"
    )
    print(
        f"Perplexidade PT-BR:      {metrics['perplexity_ptbr']:.4f} "
        f"(tokens={metrics['perplexity_scored_tokens']}, "
        f"NLL={metrics['perplexity_nll_sum']:.4f})"
    )
    print(
        f"Treino até o fim:        SIM — "
        f"{report['campaign']['completed_steps']}/{report['campaign']['planned_steps']} passos"
    )
    print("=" * 80)
    print("JSON_RESULT=" + json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
