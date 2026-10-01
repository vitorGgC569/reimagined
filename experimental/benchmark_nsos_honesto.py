#!/usr/bin/env python3
"""
benchmark_nsos_honesto.py

Suíte local, reproduzível e sem cherry-picking para avaliar nsos_ext.JambaModel.

O que mede:
- QAR / associative recall com vocabulários disjuntos.
- Selective copy.
- Generalização fora da distribuição por comprimento e quantidade de pares.
- Perplexidade teacher-forced.
- Acurácia greedy e exact match.
- Throughput de prefill e geração greedy ponta a ponta.
- Múltiplas seeds, média, desvio-padrão e IC95%.
- Ablations de OxtaMem, QAT e estado do otimizador.
- Separação train/validation/test por hash, impedindo repetição exata entre splits.
- Checkpoints, estado do otimizador, logs CSV/JSON e metadados do ambiente.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import gc
import hashlib
import json
import math
import os
import pickle
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Tokens e espaços disjuntos
# ---------------------------------------------------------------------------

PAD = 0
QUERY = 1
SELECT = 2
SEP = 3
EOS = 4

KEY_TOKENS = tuple(range(10, 50))
VALUE_TOKENS = tuple(range(50, 90))
NOISE_TOKENS = tuple(range(90, 160))
COPY_TOKENS = tuple(range(20, 90))

DEFAULT_VOCAB_SIZE = 256


# ---------------------------------------------------------------------------
# Perfis
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Profile:
    name: str
    seeds: tuple[int, ...]
    qar_steps: int
    copy_steps: int
    batch_size: int
    eval_every: int
    val_samples: int
    test_samples: int
    perf_repeats: int
    perf_lengths: tuple[int, ...]


PROFILES: dict[str, Profile] = {
    "smoke": Profile(
        name="smoke",
        seeds=(42,),
        qar_steps=300,
        copy_steps=300,
        batch_size=8,
        eval_every=100,
        val_samples=64,
        test_samples=128,
        perf_repeats=8,
        perf_lengths=(32, 64, 128),
    ),
    "standard": Profile(
        name="standard",
        seeds=(42, 123, 2026),
        qar_steps=4_000,
        copy_steps=4_000,
        batch_size=16,
        eval_every=250,
        val_samples=256,
        test_samples=512,
        perf_repeats=30,
        perf_lengths=(32, 64, 128, 256, 512),
    ),
    "full": Profile(
        name="full",
        seeds=(42, 123, 2026, 31415, 27182),
        qar_steps=10_000,
        copy_steps=10_000,
        batch_size=32,
        eval_every=500,
        val_samples=1_000,
        test_samples=2_000,
        perf_repeats=100,
        perf_lengths=(32, 64, 128, 256, 512, 1024),
    ),
}


# ---------------------------------------------------------------------------
# Variantes / ablations
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class VariantSpec:
    name: str
    auxiliary_memory_enabled: bool
    progressive_qat_enabled: bool
    quantized_precision_bits: int
    optimizer_state_bits: int
    model_config: dict[str, Any] = dataclasses.field(default_factory=dict)


BUILTIN_VARIANTS: dict[str, VariantSpec] = {
    "float_mem_opt32": VariantSpec(
        name="float_mem_opt32",
        auxiliary_memory_enabled=True,
        progressive_qat_enabled=False,
        quantized_precision_bits=2,
        optimizer_state_bits=32,
    ),
    "float_nomem_opt32": VariantSpec(
        name="float_nomem_opt32",
        auxiliary_memory_enabled=False,
        progressive_qat_enabled=False,
        quantized_precision_bits=2,
        optimizer_state_bits=32,
    ),
    "float_mem_opt4": VariantSpec(
        name="float_mem_opt4",
        auxiliary_memory_enabled=True,
        progressive_qat_enabled=False,
        quantized_precision_bits=2,
        optimizer_state_bits=4,
    ),
    "qat2_mem_opt32": VariantSpec(
        name="qat2_mem_opt32",
        auxiliary_memory_enabled=True,
        progressive_qat_enabled=True,
        quantized_precision_bits=2,
        optimizer_state_bits=32,
    ),
}


# ---------------------------------------------------------------------------
# Tarefas
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Scenario:
    name: str
    params: dict[str, int]


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    name: str
    train_params: dict[str, int]
    validation: Scenario
    tests: tuple[Scenario, ...]


TASKS: dict[str, TaskSpec] = {
    "qar": TaskSpec(
        name="qar",
        train_params={"n_kv": 8, "seq_len": 64},
        validation=Scenario("id", {"n_kv": 8, "seq_len": 64}),
        tests=(
            Scenario("id", {"n_kv": 8, "seq_len": 64}),
            Scenario("length_2x", {"n_kv": 8, "seq_len": 128}),
            Scenario("length_4x", {"n_kv": 8, "seq_len": 256}),
            Scenario("pairs_2x", {"n_kv": 16, "seq_len": 128}),
            Scenario("pairs_2x_length_4x", {"n_kv": 16, "seq_len": 256}),
        ),
    ),
    "copy": TaskSpec(
        name="copy",
        train_params={"n_selected": 4, "seq_len": 64},
        validation=Scenario("id", {"n_selected": 4, "seq_len": 64}),
        tests=(
            Scenario("id", {"n_selected": 4, "seq_len": 64}),
            Scenario("length_2x", {"n_selected": 4, "seq_len": 128}),
            Scenario("length_4x", {"n_selected": 4, "seq_len": 256}),
            Scenario("selected_2x", {"n_selected": 8, "seq_len": 128}),
            Scenario("selected_2x_length_4x", {"n_selected": 8, "seq_len": 256}),
        ),
    ),
}


# ---------------------------------------------------------------------------
# Utilitários gerais
# ---------------------------------------------------------------------------

def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def stable_int_seed(*parts: Any) -> int:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") & 0x7FFFFFFF


def sample_hash(prompt: Sequence[int], answer: Sequence[int]) -> str:
    payload = bytes(int(x) & 0xFF for x in (*prompt, 255, *answer))
    return hashlib.sha256(payload).hexdigest()


def split_bucket(prompt: Sequence[int], answer: Sequence[int]) -> int:
    return int(sample_hash(prompt, answer)[:8], 16) % 1000


def belongs_to_split(prompt: Sequence[int], answer: Sequence[int], split: str) -> bool:
    bucket = split_bucket(prompt, answer)
    if split == "train":
        return bucket < 900
    if split == "validation":
        return 900 <= bucket < 950
    if split == "test":
        return 950 <= bucket < 1000
    raise ValueError(f"Split inválido: {split}")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def command_output(command: list[str], timeout: int = 10) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def current_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore
        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    except Exception:
        return None


def current_gpu_memory_mb() -> float | None:
    text = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout=5,
    )
    if not text:
        return None
    pid = str(os.getpid())
    total = 0.0
    found = False
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2 or parts[0] != pid:
            continue
        try:
            total += float(parts[1])
            found = True
        except ValueError:
            pass
    return total if found else None


def t_critical_95(df: int) -> float:
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    if df <= 0:
        return float("nan")
    return table.get(df, 1.96)


def mean_std_ci(values: Sequence[float]) -> tuple[float, float, float]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return float("nan"), float("nan"), float("nan")
    mean = statistics.fmean(clean)
    if len(clean) == 1:
        return mean, 0.0, float("nan")
    std = statistics.stdev(clean)
    ci = t_critical_95(len(clean) - 1) * std / math.sqrt(len(clean))
    return mean, std, ci


# ---------------------------------------------------------------------------
# Importação do módulo nativo
# ---------------------------------------------------------------------------

def find_nsos_paths(cli_path: str | None) -> list[Path]:
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()
    candidates: list[Path] = []

    env_path = os.environ.get("NSOS_EXT_PATH")
    for raw in (cli_path, env_path):
        if raw:
            candidates.append(Path(raw).expanduser().resolve())

    candidates.extend(
        [
            here / "OXN" / "build" / "Release",
            here.parent / "OXN" / "build" / "Release",
            cwd / "OXN" / "build" / "Release",
            cwd / "build" / "Release",
            here,
            cwd,
        ]
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def import_nsos_ext(cli_path: str | None):
    searched = find_nsos_paths(cli_path)
    for path in searched:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    try:
        import nsos_ext  # type: ignore
        return nsos_ext, searched
    except Exception as exc:
        lines = "\n".join(f"  - {p}" for p in searched)
        raise RuntimeError(
            "Não foi possível importar nsos_ext.\n"
            "Caminhos pesquisados:\n"
            f"{lines}\n"
            "Use --nsos-path CAMINHO ou defina NSOS_EXT_PATH.\n"
            f"Erro original: {exc}"
        ) from exc


def try_seed_native(nsos_ext: Any, seed: int) -> str | None:
    for name in ("set_seed", "manual_seed", "seed"):
        fn = getattr(nsos_ext, name, None)
        if callable(fn):
            try:
                fn(int(seed))
                return name
            except Exception:
                continue
    return None


def resolve_device(nsos_ext: Any, requested: str) -> tuple[Any | None, str]:
    enum = getattr(nsos_ext, "Device", None)
    if enum is None:
        return None, "constructor_default"

    if requested == "cpu":
        names = ("CPU",)
    elif requested in ("cuda", "gpu"):
        names = ("CUDA", "GPU")
    else:
        names = ("CUDA", "GPU", "CPU")

    for name in names:
        if hasattr(enum, name):
            return getattr(enum, name), name
    return None, "constructor_default"


# ---------------------------------------------------------------------------
# Modelo e trainer
# ---------------------------------------------------------------------------

def apply_config_field(config: Any, field: str, value: Any) -> None:
    if not hasattr(config, field):
        raise AttributeError(
            f"ModelConfig não expõe o campo '{field}'. "
            "Use apenas nomes reais do seu binding PyBind11."
        )
    setattr(config, field, value)


def build_model(
    nsos_ext: Any,
    variant: VariantSpec,
    d_model: int,
    num_layers: int,
    vocab_size: int,
    requested_device: str,
):
    device, device_name = resolve_device(nsos_ext, requested_device)

    if variant.model_config:
        if not hasattr(nsos_ext, "ModelConfig"):
            raise RuntimeError("Esta variante exige nsos_ext.ModelConfig.")
        config = nsos_ext.ModelConfig()

        for field, value in (("d_model", d_model), ("num_layers", num_layers)):
            apply_config_field(config, field, value)

        if hasattr(config, "vocab_size"):
            setattr(config, "vocab_size", vocab_size)

        for field, value in variant.model_config.items():
            apply_config_field(config, field, value)

        model = nsos_ext.JambaModel(config)
    else:
        model = nsos_ext.JambaModel(int(num_layers), int(d_model), int(vocab_size))
        device_name = "CPU"

    model.set_training_mode(True)
    return model, device_name


def configure_trainer(
    nsos_ext: Any,
    model: Any,
    variant: VariantSpec,
    learning_rate: float,
    warmup_steps: int,
    max_grad_norm: float,
    weight_decay: float,
    first_token_loss_scale: float,
    eos_loss_scale: float,
    moe_aux_loss_scale: float,
    logit_l2_beta: float,
):
    trainer = nsos_ext.Trainer(model, learning_rate=float(learning_rate))

    settings = {
        "optimizer_state_bits": int(variant.optimizer_state_bits),
        "warmup_steps": int(warmup_steps),
        "max_grad_norm": float(max_grad_norm),
        "weight_decay": float(weight_decay),
        "first_token_loss_scale": float(first_token_loss_scale),
        "eos_loss_scale": float(eos_loss_scale),
        "moe_aux_loss_scale": float(moe_aux_loss_scale),
        "logit_l2_beta": float(logit_l2_beta),
    }
    for field, value in settings.items():
        if not hasattr(trainer, field):
            raise AttributeError(f"Trainer não expõe '{field}'.")
        setattr(trainer, field, value)

    if not hasattr(nsos_ext, "TrainPhaseScheduler"):
        raise RuntimeError("nsos_ext.TrainPhaseScheduler não está disponível.")
    scheduler = nsos_ext.TrainPhaseScheduler()
    scheduler.progressive_qat_enabled = bool(variant.progressive_qat_enabled)
    scheduler.quantized_precision_bits = int(variant.quantized_precision_bits)
    scheduler.auxiliary_memory_enabled = bool(variant.auxiliary_memory_enabled)
    trainer.configure_progressive_qat(scheduler)

    return trainer


def parameter_count(model: Any) -> int | None:
    try:
        params = model.parameters()
    except Exception:
        return None

    total = 0
    found = False
    for param in params:
        numel_attr = getattr(param, "numel", None)
        if callable(numel_attr):
            try:
                total += int(numel_attr())
                found = True
                continue
            except Exception:
                pass

        shape = getattr(param, "shape", None)
        if callable(shape):
            try:
                shape = shape()
            except Exception:
                shape = None
        if shape is not None:
            try:
                dims = [int(x) for x in shape]
                total += math.prod(dims)
                found = True
                continue
            except Exception:
                pass

        try:
            arr = tensor_to_numpy(param)
            total += int(arr.size)
            found = True
        except Exception:
            pass

    return total if found else None


def reset_model_state(model: Any) -> str | None:
    for name in (
        "reset_state",
        "reset_memory",
        "clear_memory",
        "reset_cache",
        "clear_cache",
    ):
        fn = getattr(model, name, None)
        if callable(fn):
            try:
                fn()
                return name
            except Exception:
                continue
    return None


def synchronize_backend(nsos_ext: Any, model: Any | None = None) -> str | None:
    for owner, names in (
        (model, ("synchronize", "sync")),
        (nsos_ext, ("synchronize", "cuda_synchronize", "device_synchronize")),
    ):
        if owner is None:
            continue
        for name in names:
            fn = getattr(owner, name, None)
            if callable(fn):
                try:
                    fn()
                    return name
                except Exception:
                    continue
    return None


def clear_backend_cache(nsos_ext: Any) -> None:
    for name in ("empty_cache", "clear_cache", "cuda_empty_cache"):
        fn = getattr(nsos_ext, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Conversão da saída nativa e predição
# ---------------------------------------------------------------------------

def tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value

    if isinstance(value, tuple) and value:
        value = value[0]

    for name in ("to_cpu_numpy", "numpy", "to_numpy"):
        fn = getattr(value, name, None)
        if callable(fn):
            try:
                arr = fn()
                return np.asarray(arr)
            except Exception:
                pass

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return np.asarray(tolist())
        except Exception:
            pass

    try:
        arr = np.asarray(value)
        if arr.dtype != object:
            return arr
    except Exception:
        pass

    raise TypeError(
        "Não foi possível converter a saída de forward_ids para NumPy. "
        f"Tipo recebido: {type(value)!r}. "
        "Exponha logits por numpy(), to_numpy(), to_cpu_numpy() ou tolist()."
    )


def last_token_logits(model: Any, prompt: Sequence[int], vocab_size: int) -> np.ndarray:
    output = model.forward_ids([int(x) for x in prompt])
    arr = tensor_to_numpy(output)

    if arr.ndim == 0:
        raise ValueError("forward_ids retornou um escalar, não logits.")
    if arr.ndim == 1:
        logits = arr
    elif arr.ndim == 2:
        logits = arr[-1]
    else:
        logits = arr.reshape((-1, arr.shape[-1]))[-1]

    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape[-1] != vocab_size:
        raise ValueError(
            "A última dimensão da saída não coincide com vocab_size. "
            f"shape={arr.shape}, última dimensão={logits.shape[-1]}, "
            f"vocab_size={vocab_size}. "
            "Isso pode indicar que forward_ids retorna hidden states em vez de logits."
        )
    return logits


def log_softmax_nll(logits: np.ndarray, target: int) -> float:
    maximum = float(np.max(logits))
    logsumexp = maximum + math.log(float(np.exp(logits - maximum).sum()))
    return logsumexp - float(logits[int(target)])


# ---------------------------------------------------------------------------
# Geração dos datasets
# ---------------------------------------------------------------------------

def make_qar_sample_raw(
    rng: random.Random,
    n_kv: int,
    seq_len: int,
) -> tuple[list[int], list[int]]:
    if n_kv > len(KEY_TOKENS) or n_kv > len(VALUE_TOKENS):
        raise ValueError("n_kv excede o vocabulário de chaves/valores.")
    minimum = 2 * n_kv + 2
    if seq_len < minimum:
        raise ValueError(f"seq_len precisa ser >= {minimum} para n_kv={n_kv}.")

    keys = rng.sample(KEY_TOKENS, n_kv)
    values = rng.sample(VALUE_TOKENS, n_kv)
    mapping = dict(zip(keys, values))

    prompt: list[int] = []
    for key, value in zip(keys, values):
        prompt.extend((key, value))

    noise_len = seq_len - len(prompt) - 2
    prompt.extend(rng.choice(NOISE_TOKENS) for _ in range(noise_len))

    query_key = rng.choice(keys)
    prompt.extend((QUERY, query_key))
    answer = [mapping[query_key]]
    return prompt, answer


def make_copy_sample_raw(
    rng: random.Random,
    n_selected: int,
    seq_len: int,
) -> tuple[list[int], list[int]]:
    n_items = seq_len - 1 - n_selected
    if n_items < n_selected or n_selected <= 0:
        raise ValueError("Configuração inválida para selective copy.")

    selected_positions = sorted(rng.sample(range(n_items), n_selected))
    selected_set = set(selected_positions)
    payload = [rng.choice(COPY_TOKENS) for _ in range(n_items)]

    prompt: list[int] = []
    for index, token in enumerate(payload):
        if index in selected_set:
            prompt.append(SELECT)
        prompt.append(token)
    prompt.append(QUERY)

    answer = [payload[index] for index in selected_positions] + [EOS]
    assert len(prompt) == seq_len
    return prompt, answer


def make_raw_sample(
    task_name: str,
    rng: random.Random,
    params: dict[str, int],
) -> tuple[list[int], list[int]]:
    if task_name == "qar":
        return make_qar_sample_raw(rng, **params)
    if task_name == "copy":
        return make_copy_sample_raw(rng, **params)
    raise ValueError(f"Tarefa desconhecida: {task_name}")


def make_split_sample(
    task_name: str,
    rng: random.Random,
    params: dict[str, int],
    split: str,
) -> tuple[list[int], list[int]]:
    for _ in range(100_000):
        prompt, answer = make_raw_sample(task_name, rng, params)
        if belongs_to_split(prompt, answer, split):
            return prompt, answer
    raise RuntimeError("Não foi possível gerar uma amostra para o split solicitado.")


def build_fixed_dataset(
    task_name: str,
    params: dict[str, int],
    split: str,
    count: int,
    seed: int,
) -> list[tuple[list[int], list[int]]]:
    rng = random.Random(seed)
    dataset: list[tuple[list[int], list[int]]] = []
    seen: set[str] = set()

    while len(dataset) < count:
        prompt, answer = make_split_sample(task_name, rng, params, split)
        digest = sample_hash(prompt, answer)
        if digest in seen:
            continue
        seen.add(digest)
        dataset.append((prompt, answer))

    return dataset


def assert_split_disjointness(
    task_name: str,
    params: dict[str, int],
    seed: int,
    count: int = 256,
) -> dict[str, int]:
    hashes: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        rng = random.Random(stable_int_seed(seed, task_name, split, "leakcheck"))
        values: set[str] = set()
        while len(values) < count:
            prompt, answer = make_split_sample(task_name, rng, params, split)
            values.add(sample_hash(prompt, answer))
        hashes[split] = values

    assert hashes["train"].isdisjoint(hashes["validation"])
    assert hashes["train"].isdisjoint(hashes["test"])
    assert hashes["validation"].isdisjoint(hashes["test"])
    return {name: len(values) for name, values in hashes.items()}


# ---------------------------------------------------------------------------
# Treino e avaliação
# ---------------------------------------------------------------------------

class TrainerAdapter:
    def __init__(self, trainer: Any):
        self.trainer = trainer
        self.batch_enabled = callable(
            getattr(trainer, "train_supervised_batch", None)
        )
        self.batch_failure_reported = False

    def train_batch(
        self,
        prompts: list[list[int]],
        answers: list[list[int]],
    ) -> float:
        if self.batch_enabled:
            try:
                value = self.trainer.train_supervised_batch(prompts, answers)
                if value is None:
                    return float("nan")
                return float(value)
            except Exception as exc:
                self.batch_enabled = False
                if not self.batch_failure_reported:
                    print(
                        "[aviso] train_supervised_batch falhou; "
                        f"usando train_step individual. Motivo: {exc}"
                    )
                    self.batch_failure_reported = True

        losses: list[float] = []
        for prompt, answer in zip(prompts, answers):
            value = self.trainer.train_step(prompt, answer)
            if value is not None:
                losses.append(float(value))
        return statistics.fmean(losses) if losses else float("nan")


def evaluate_dataset(
    model: Any,
    dataset: Sequence[tuple[list[int], list[int]]],
    vocab_size: int,
) -> dict[str, float | int | str | None]:
    model.set_training_mode(False)

    exact = 0
    greedy_correct = 0
    greedy_total = 0
    nll_total = 0.0
    nll_tokens = 0
    reset_method: str | None = None

    started = time.perf_counter()

    for prompt, answer in dataset:
        reset_method = reset_model_state(model) or reset_method

        gold_context = list(prompt)
        greedy_context = list(prompt)
        greedy_sequence: list[int] = []

        for gold_token in answer:
            gold_logits = last_token_logits(model, gold_context, vocab_size)
            nll_total += log_softmax_nll(gold_logits, gold_token)
            nll_tokens += 1
            gold_context.append(int(gold_token))

            greedy_logits = last_token_logits(model, greedy_context, vocab_size)
            prediction = int(np.argmax(greedy_logits))
            greedy_sequence.append(prediction)
            greedy_correct += int(prediction == gold_token)
            greedy_total += 1
            greedy_context.append(prediction)

        exact += int(greedy_sequence == answer)

    elapsed = time.perf_counter() - started
    mean_nll = nll_total / max(1, nll_tokens)
    ppl = math.exp(min(mean_nll, 80.0))

    return {
        "samples": len(dataset),
        "exact_match": exact / max(1, len(dataset)),
        "greedy_token_accuracy": greedy_correct / max(1, greedy_total),
        "mean_nll": mean_nll,
        "perplexity": ppl,
        "eval_seconds": elapsed,
        "samples_per_second": len(dataset) / max(elapsed, 1e-12),
        "reset_method": reset_method,
    }


def save_training_checkpoint(
    trainer: Any,
    model: Any,
    state_path: Path,
    model_path: Path,
    rng_path: Path,
    rng: random.Random,
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    saved_native_state = False

    fn = getattr(trainer, "save_training_state", None)
    if callable(fn):
        try:
            fn(str(state_path), str(model_path))
            saved_native_state = True
        except Exception as exc:
            print(f"[aviso] save_training_state falhou: {exc}")

    if not saved_native_state:
        model.save(str(model_path))

    with rng_path.open("wb") as f:
        pickle.dump(rng.getstate(), f)


def load_training_checkpoint(
    trainer: Any,
    model: Any,
    state_path: Path,
    model_path: Path,
    rng_path: Path,
    rng: random.Random,
) -> int:
    loaded_native_state = False
    fn = getattr(trainer, "load_training_state", None)

    if callable(fn) and state_path.exists() and model_path.exists():
        try:
            fn(str(state_path), str(model_path))
            loaded_native_state = True
        except Exception as exc:
            print(f"[aviso] load_training_state falhou: {exc}")

    if not loaded_native_state and model_path.exists():
        model.load(str(model_path))

    if rng_path.exists():
        with rng_path.open("rb") as f:
            rng.setstate(pickle.load(f))

    return int(getattr(trainer, "global_step_count", 0))


def benchmark_prefill(
    nsos_ext: Any,
    model: Any,
    vocab_size: int,
    lengths: Sequence[int],
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    model.set_training_mode(False)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for length in lengths:
        prompt = [rng.randrange(10, min(vocab_size, 160)) for _ in range(length)]

        for _ in range(5):
            model.forward_ids(prompt)
        synchronize_backend(nsos_ext, model)

        rss_before = current_rss_mb()
        gpu_before = current_gpu_memory_mb()

        start = time.perf_counter()
        for _ in range(repeats):
            model.forward_ids(prompt)
        synchronize_method = synchronize_backend(nsos_ext, model)
        elapsed = time.perf_counter() - start

        rss_after = current_rss_mb()
        gpu_after = current_gpu_memory_mb()

        rows.append(
            {
                "mode": "prefill",
                "context_length": length,
                "repeats": repeats,
                "elapsed_seconds": elapsed,
                "latency_ms": 1000.0 * elapsed / repeats,
                "tokens_per_second": (length * repeats) / max(elapsed, 1e-12),
                "rss_before_mb": rss_before,
                "rss_after_mb": rss_after,
                "gpu_before_mb": gpu_before,
                "gpu_after_mb": gpu_after,
                "synchronize_method": synchronize_method,
            }
        )
    return rows


def benchmark_greedy_generation(
    nsos_ext: Any,
    model: Any,
    vocab_size: int,
    context_length: int,
    generated_tokens: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    model.set_training_mode(False)
    rng = random.Random(seed)
    base_prompt = [
        rng.randrange(10, min(vocab_size, 160)) for _ in range(context_length)
    ]

    for _ in range(2):
        context = list(base_prompt)
        for _ in range(min(generated_tokens, 4)):
            logits = last_token_logits(model, context, vocab_size)
            context.append(int(np.argmax(logits)))
    synchronize_backend(nsos_ext, model)

    start = time.perf_counter()
    total_generated = 0
    for _ in range(repeats):
        context = list(base_prompt)
        for _ in range(generated_tokens):
            logits = last_token_logits(model, context, vocab_size)
            context.append(int(np.argmax(logits)))
            total_generated += 1
    synchronize_method = synchronize_backend(nsos_ext, model)
    elapsed = time.perf_counter() - start

    return {
        "mode": "greedy_end_to_end",
        "context_length": context_length,
        "generated_tokens_per_repeat": generated_tokens,
        "repeats": repeats,
        "elapsed_seconds": elapsed,
        "latency_ms_per_generated_token": 1000.0 * elapsed / max(total_generated, 1),
        "generated_tokens_per_second": total_generated / max(elapsed, 1e-12),
        "synchronize_method": synchronize_method,
    }


def run_single_training(
    *,
    nsos_ext: Any,
    args: argparse.Namespace,
    profile: Profile,
    variant: VariantSpec,
    task: TaskSpec,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    run_name = f"{variant.name}__{task.name}__seed{seed}"
    run_dir = output_dir / "runs" / run_name
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 88)
    print(f"RUN: {run_name}")
    print("=" * 88)

    random.seed(seed)
    np.random.seed(seed)
    native_seed_method = try_seed_native(nsos_ext, seed)

    model, device_name = build_model(
        nsos_ext=nsos_ext,
        variant=variant,
        d_model=args.d_model,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        requested_device=args.device,
    )
    trainer = configure_trainer(
        nsos_ext=nsos_ext,
        model=model,
        variant=variant,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        first_token_loss_scale=args.first_token_loss_scale,
        eos_loss_scale=args.eos_loss_scale,
        moe_aux_loss_scale=args.moe_aux_loss_scale,
        logit_l2_beta=args.logit_l2_beta,
    )
    adapter = TrainerAdapter(trainer)

    params = parameter_count(model)
    print(f"Device: {device_name}")
    print(f"Parâmetros detectados: {params if params is not None else 'indisponível'}")
    print(
        f"QAT={variant.progressive_qat_enabled} | "
        f"Memória={variant.auxiliary_memory_enabled} | "
        f"Optimizer bits={variant.optimizer_state_bits}"
    )

    val_seed = stable_int_seed(seed, variant.name, task.name, "validation")
    val_dataset = build_fixed_dataset(
        task_name=task.name,
        params=task.validation.params,
        split="validation",
        count=profile.val_samples,
        seed=val_seed,
    )

    steps_target = profile.qar_steps if task.name == "qar" else profile.copy_steps
    train_rng = random.Random(
        stable_int_seed(seed, variant.name, task.name, "train_stream")
    )

    state_path = checkpoint_dir / "last.state"
    last_model_path = checkpoint_dir / "last_model.bin"
    rng_path = checkpoint_dir / "last_rng.pkl"
    best_model_path = checkpoint_dir / "best_model.bin"

    start_step = 0
    if args.resume and (state_path.exists() or last_model_path.exists()):
        start_step = load_training_checkpoint(
            trainer, model, state_path, last_model_path, rng_path, train_rng
        )
        print(f"Retomando do passo global {start_step}.")

    best_exact = -1.0
    best_nll = float("inf")
    best_step = start_step
    training_started = time.perf_counter()
    interval_started = training_started
    interval_tokens = 0
    interval_samples = 0

    log_csv = run_dir / "training_log.csv"

    for step in range(start_step + 1, steps_target + 1):
        prompts: list[list[int]] = []
        answers: list[list[int]] = []

        for _ in range(profile.batch_size):
            prompt, answer = make_split_sample(
                task_name=task.name,
                rng=train_rng,
                params=task.train_params,
                split="train",
            )
            prompts.append(prompt)
            answers.append(answer)
            interval_tokens += len(prompt) + len(answer)
            interval_samples += 1

        loss = adapter.train_batch(prompts, answers)

        should_eval = (
            step == 1
            or step % profile.eval_every == 0
            or step == steps_target
        )
        if not should_eval:
            continue

        synchronize_backend(nsos_ext, model)
        now = time.perf_counter()
        interval_elapsed = now - interval_started
        train_tok_s = interval_tokens / max(interval_elapsed, 1e-12)
        train_samples_s = interval_samples / max(interval_elapsed, 1e-12)

        metrics = evaluate_dataset(model, val_dataset, args.vocab_size)
        model.set_training_mode(True)

        exact = float(metrics["exact_match"])
        nll = float(metrics["mean_nll"])
        improved = exact > best_exact or (
            math.isclose(exact, best_exact) and nll < best_nll
        )
        if improved:
            best_exact = exact
            best_nll = nll
            best_step = step
            model.save(str(best_model_path))

        save_training_checkpoint(
            trainer=trainer,
            model=model,
            state_path=state_path,
            model_path=last_model_path,
            rng_path=rng_path,
            rng=train_rng,
        )

        row = {
            "run": run_name,
            "variant": variant.name,
            "task": task.name,
            "seed": seed,
            "step": step,
            "global_step_count": int(getattr(trainer, "global_step_count", step)),
            "train_loss": loss,
            "train_tokens_per_second": train_tok_s,
            "train_samples_per_second": train_samples_s,
            "val_exact_match": metrics["exact_match"],
            "val_greedy_token_accuracy": metrics["greedy_token_accuracy"],
            "val_mean_nll": metrics["mean_nll"],
            "val_perplexity": metrics["perplexity"],
            "val_samples_per_second": metrics["samples_per_second"],
            "best_step": best_step,
            "best_exact_match": best_exact,
            "best_mean_nll": best_nll,
            "rss_mb": current_rss_mb(),
            "gpu_memory_mb": current_gpu_memory_mb(),
        }
        append_csv(log_csv, row)

        print(
            f"step={step:6d}/{steps_target} "
            f"loss={loss:9.5f} "
            f"val_exact={exact:7.3%} "
            f"val_ppl={float(metrics['perplexity']):9.4f} "
            f"train_tok/s={train_tok_s:10.1f}"
        )

        interval_started = time.perf_counter()
        interval_tokens = 0
        interval_samples = 0

    training_seconds = time.perf_counter() - training_started

    if not best_model_path.exists():
        model.save(str(best_model_path))
    model.load(str(best_model_path))
    model.set_training_mode(False)

    test_results: list[dict[str, Any]] = []
    for scenario in task.tests:
        scenario_seed = stable_int_seed(
            seed, variant.name, task.name, scenario.name, "test"
        )
        dataset = build_fixed_dataset(
            task_name=task.name,
            params=scenario.params,
            split="test",
            count=profile.test_samples,
            seed=scenario_seed,
        )
        metrics = evaluate_dataset(model, dataset, args.vocab_size)
        row = {
            "variant": variant.name,
            "task": task.name,
            "seed": seed,
            "scenario": scenario.name,
            "scenario_params": scenario.params,
            **metrics,
        }
        test_results.append(row)
        print(
            f"TEST {scenario.name:<24} "
            f"exact={float(metrics['exact_match']):7.3%} "
            f"token_acc={float(metrics['greedy_token_accuracy']):7.3%} "
            f"ppl={float(metrics['perplexity']):9.4f}"
        )

    perf_rows = benchmark_prefill(
        nsos_ext=nsos_ext,
        model=model,
        vocab_size=args.vocab_size,
        lengths=profile.perf_lengths,
        repeats=profile.perf_repeats,
        seed=stable_int_seed(seed, variant.name, task.name, "prefill"),
    )
    generation_row = benchmark_greedy_generation(
        nsos_ext=nsos_ext,
        model=model,
        vocab_size=args.vocab_size,
        context_length=min(128, max(profile.perf_lengths)),
        generated_tokens=32,
        repeats=max(2, profile.perf_repeats // 10),
        seed=stable_int_seed(seed, variant.name, task.name, "generation"),
    )

    run_result = {
        "run_name": run_name,
        "variant": dataclasses.asdict(variant),
        "task": dataclasses.asdict(task),
        "seed": seed,
        "native_seed_method": native_seed_method,
        "device": device_name,
        "parameter_count": params,
        "training_seconds": training_seconds,
        "steps_target": steps_target,
        "best_step": best_step,
        "best_validation_exact_match": best_exact,
        "best_validation_mean_nll": best_nll,
        "test_results": test_results,
        "performance": perf_rows + [generation_row],
        "state_reset_method_detected": reset_model_state(model),
        "checkpoint": str(best_model_path),
    }
    write_json(run_dir / "result.json", run_result)

    del adapter, trainer, model
    gc.collect()
    clear_backend_cache(nsos_ext)

    return run_result


# ---------------------------------------------------------------------------
# Agregação
# ---------------------------------------------------------------------------

def aggregate_results(
    run_results: Sequence[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    for run in run_results:
        variant = run["variant"]["name"]
        task = run["task"]["name"]
        for result in run["test_results"]:
            key = (variant, task, result["scenario"])
            grouped.setdefault(key, []).append(result)

    summary_rows: list[dict[str, Any]] = []
    for (variant, task, scenario), rows in sorted(grouped.items()):
        exact_mean, exact_std, exact_ci = mean_std_ci(
            [float(r["exact_match"]) for r in rows]
        )
        token_mean, token_std, token_ci = mean_std_ci(
            [float(r["greedy_token_accuracy"]) for r in rows]
        )
        ppl_mean, ppl_std, ppl_ci = mean_std_ci(
            [float(r["perplexity"]) for r in rows]
        )
        nll_mean, nll_std, nll_ci = mean_std_ci(
            [float(r["mean_nll"]) for r in rows]
        )
        summary_rows.append(
            {
                "variant": variant,
                "task": task,
                "scenario": scenario,
                "n_seeds": len(rows),
                "exact_match_mean": exact_mean,
                "exact_match_std": exact_std,
                "exact_match_ci95": exact_ci,
                "greedy_token_accuracy_mean": token_mean,
                "greedy_token_accuracy_std": token_std,
                "greedy_token_accuracy_ci95": token_ci,
                "perplexity_mean": ppl_mean,
                "perplexity_std": ppl_std,
                "perplexity_ci95": ppl_ci,
                "mean_nll_mean": nll_mean,
                "mean_nll_std": nll_std,
                "mean_nll_ci95": nll_ci,
            }
        )

    summary_csv = output_dir / "summary.csv"
    if summary_csv.exists():
        summary_csv.unlink()
    for row in summary_rows:
        append_csv(summary_csv, row)
    write_json(output_dir / "summary.json", summary_rows)

    return summary_rows


def aggregate_performance(
    run_results: Sequence[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[float]] = {}

    for run in run_results:
        variant = run["variant"]["name"]
        task = run["task"]["name"]
        for row in run["performance"]:
            mode = row["mode"]
            if mode == "prefill":
                length = int(row["context_length"])
                value = float(row["tokens_per_second"])
            else:
                length = int(row["context_length"])
                value = float(row["generated_tokens_per_second"])
            grouped.setdefault((variant, task, mode, length), []).append(value)

    rows: list[dict[str, Any]] = []
    for (variant, task, mode, length), values in sorted(grouped.items()):
        mean, std, ci = mean_std_ci(values)
        rows.append(
            {
                "variant": variant,
                "task": task,
                "mode": mode,
                "context_length": length,
                "n_seeds": len(values),
                "tokens_per_second_mean": mean,
                "tokens_per_second_std": std,
                "tokens_per_second_ci95": ci,
            }
        )

    path = output_dir / "performance_summary.csv"
    if path.exists():
        path.unlink()
    for row in rows:
        append_csv(path, row)
    write_json(output_dir / "performance_summary.json", rows)
    return rows


def print_primary_summary(rows: Sequence[dict[str, Any]]) -> None:
    print("\n" + "=" * 112)
    print("RESUMO AGREGADO — média ± desvio-padrão; IC95% salvo no CSV")
    print("=" * 112)
    print(
        f"{'Variante':<24} {'Tarefa':<8} {'Cenário':<24} "
        f"{'Exact match':>18} {'Token acc':>18} {'PPL':>18}"
    )
    print("-" * 112)
    for row in rows:
        exact = (
            f"{100*row['exact_match_mean']:.2f}% "
            f"± {100*row['exact_match_std']:.2f}"
        )
        token = (
            f"{100*row['greedy_token_accuracy_mean']:.2f}% "
            f"± {100*row['greedy_token_accuracy_std']:.2f}"
        )
        ppl = f"{row['perplexity_mean']:.4f} ± {row['perplexity_std']:.4f}"
        print(
            f"{row['variant']:<24} {row['task']:<8} {row['scenario']:<24} "
            f"{exact:>18} {token:>18} {ppl:>18}"
        )


def fairness_report(
    run_results: Sequence[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    by_variant: dict[str, list[int]] = {}
    reset_methods: set[str] = set()
    missing_reset = False

    for run in run_results:
        name = run["variant"]["name"]
        count = run.get("parameter_count")
        if isinstance(count, int):
            by_variant.setdefault(name, []).append(count)
        method = run.get("state_reset_method_detected")
        if method:
            reset_methods.add(str(method))
        else:
            missing_reset = True

    variant_counts = {
        name: int(round(statistics.fmean(values)))
        for name, values in by_variant.items()
        if values
    }

    warning_messages: list[str] = []
    if variant_counts:
        minimum = min(variant_counts.values())
        maximum = max(variant_counts.values())
        if minimum > 0 and (maximum - minimum) / minimum > 0.05:
            warning_messages.append(
                "As variantes diferem em mais de 5% no número de parâmetros. "
                "Não atribua diferenças apenas à arquitetura."
            )

    if missing_reset:
        warning_messages.append(
            "Nenhum método público de reset de estado/memória foi detectado em "
            "pelo menos uma run. Confirme que forward_ids é stateless por chamada "
            "ou exponha reset_state/reset_memory para isolamento rigoroso."
        )

    architecture_configs = {
        json.dumps(run["variant"].get("model_config", {}), sort_keys=True)
        for run in run_results
    }
    if len(architecture_configs) == 1 and architecture_configs == {"{}"}:
        warning_messages.append(
            "As variantes built-in são ablations do mesmo JambaModel; este resultado "
            "não é uma comparação Transformer vs Mamba vs NSOS."
        )

    report = {
        "parameter_count_by_variant": variant_counts,
        "detected_reset_methods": sorted(reset_methods),
        "warnings": warning_messages,
    }
    write_json(output_dir / "fairness_report.json", report)
    return report


# ---------------------------------------------------------------------------
# Variantes externas
# ---------------------------------------------------------------------------

def load_variants(args: argparse.Namespace) -> list[VariantSpec]:
    if args.variants_file:
        raw = json.loads(Path(args.variants_file).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("--variants-file deve conter uma lista JSON.")
        variants: list[VariantSpec] = []
        for item in raw:
            variants.append(
                VariantSpec(
                    name=str(item["name"]),
                    auxiliary_memory_enabled=bool(
                        item.get("auxiliary_memory_enabled", True)
                    ),
                    progressive_qat_enabled=bool(
                        item.get("progressive_qat_enabled", False)
                    ),
                    quantized_precision_bits=int(
                        item.get("quantized_precision_bits", 2)
                    ),
                    optimizer_state_bits=int(
                        item.get("optimizer_state_bits", 32)
                    ),
                    model_config=dict(item.get("model_config", {})),
                )
            )
        return variants

    names = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in names if name not in BUILTIN_VARIANTS]
    if unknown:
        raise ValueError(
            f"Variantes desconhecidas: {unknown}. "
            f"Disponíveis: {sorted(BUILTIN_VARIANTS)}"
        )
    return [BUILTIN_VARIANTS[name] for name in names]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark honesto e reproduzível do nsos_ext.JambaModel."
    )
    parser.add_argument(
        "--nsos-path",
        default=None,
        help="Pasta que contém nsos_ext.pyd/.so.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="standard",
    )
    parser.add_argument(
        "--tasks",
        default="qar,copy",
        help="Lista separada por vírgulas: qar,copy",
    )
    parser.add_argument(
        "--variants",
        default="float_mem_opt32,float_nomem_opt32",
        help="Ablations built-in separadas por vírgulas.",
    )
    parser.add_argument(
        "--variants-file",
        default=None,
        help="JSON com variantes e overrides reais de ModelConfig.",
    )
    parser.add_argument("--output", default="benchmark_results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "gpu"), default="auto")

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)

    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--first-token-loss-scale", type=float, default=2.5)
    parser.add_argument("--eos-loss-scale", type=float, default=0.35)
    parser.add_argument("--moe-aux-loss-scale", type=float, default=0.01)
    parser.add_argument("--logit-l2-beta", type=float, default=0.001)

    return parser


def main() -> int:
    configure_stdout()
    parser = build_parser()
    args = parser.parse_args()

    task_names = [name.strip() for name in args.tasks.split(",") if name.strip()]
    unknown_tasks = [name for name in task_names if name not in TASKS]
    if unknown_tasks:
        parser.error(f"Tarefas desconhecidas: {unknown_tasks}")

    profile = PROFILES[args.profile]
    variants = load_variants(args)

    nsos_ext, searched_paths = import_nsos_ext(args.nsos_path)

    root = Path(args.output).expanduser().resolve()
    output_dir = root / utc_timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()
    script_sha256 = hashlib.sha256(script_path.read_bytes()).hexdigest()

    environment = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
        "nsos_ext_file": getattr(nsos_ext, "__file__", None),
        "nsos_ext_version": getattr(nsos_ext, "__version__", None),
        "searched_nsos_paths": [str(p) for p in searched_paths],
        "script_path": str(script_path),
        "script_sha256": script_sha256,
        "cwd": str(Path.cwd()),
        "git_commit_cwd": git_commit(Path.cwd()),
        "gpu_info": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }
    write_json(output_dir / "environment.json", environment)

    resolved_config = {
        "profile": dataclasses.asdict(profile),
        "tasks": [dataclasses.asdict(TASKS[name]) for name in task_names],
        "variants": [dataclasses.asdict(v) for v in variants],
        "model": {
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "vocab_size": args.vocab_size,
            "device": args.device,
        },
        "trainer": {
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "max_grad_norm": args.max_grad_norm,
            "weight_decay": args.weight_decay,
            "first_token_loss_scale": args.first_token_loss_scale,
            "eos_loss_scale": args.eos_loss_scale,
            "moe_aux_loss_scale": args.moe_aux_loss_scale,
            "logit_l2_beta": args.logit_l2_beta,
        },
        "split_policy": {
            "train": "sha256(sample) % 1000 < 900",
            "validation": "900 <= sha256(sample) % 1000 < 950",
            "test": "950 <= sha256(sample) % 1000",
        },
    }
    write_json(output_dir / "config.json", resolved_config)

    leak_checks = {}
    for task_name in task_names:
        task = TASKS[task_name]
        leak_checks[task_name] = assert_split_disjointness(
            task_name=task_name,
            params=task.train_params,
            seed=20260728,
        )
    write_json(output_dir / "split_leak_check.json", leak_checks)

    print(f"Resultados: {output_dir}")
    print(f"Perfil: {profile.name}")
    print(f"Seeds: {profile.seeds}")
    print(f"Tarefas: {task_names}")
    print(f"Variantes: {[v.name for v in variants]}")
    print("Splits verificados como disjuntos por hash.")

    run_results: list[dict[str, Any]] = []
    raw_results_path = output_dir / "raw_results.jsonl"

    for variant in variants:
        for task_name in task_names:
            task = TASKS[task_name]
            for seed in profile.seeds:
                result = run_single_training(
                    nsos_ext=nsos_ext,
                    args=args,
                    profile=profile,
                    variant=variant,
                    task=task,
                    seed=seed,
                    output_dir=output_dir,
                )
                run_results.append(result)
                append_jsonl(raw_results_path, result)

    summary_rows = aggregate_results(run_results, output_dir)
    aggregate_performance(run_results, output_dir)
    fairness = fairness_report(run_results, output_dir)
    write_json(output_dir / "all_results.json", run_results)

    print_primary_summary(summary_rows)

    if fairness["warnings"]:
        print("\nVALIDADE / LIMITAÇÕES")
        for warning in fairness["warnings"]:
            print(f"- {warning}")

    print("\nArquivos principais:")
    print(f"- {output_dir / 'summary.csv'}")
    print(f"- {output_dir / 'performance_summary.csv'}")
    print(f"- {output_dir / 'fairness_report.json'}")
    print(f"- {output_dir / 'environment.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
