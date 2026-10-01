from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .datasets import CharTokenizer, TextPair, pairs_to_token_ids


def _load_nsos_ext():
    import importlib.util
    import sys

    def _load_extension_from_path(path: Path):
        spec = importlib.util.spec_from_file_location("nsos_ext", str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["nsos_ext"] = module
        spec.loader.exec_module(module)
        return module

    ext_path = os.environ.get("NSOS_EXT_PATH")
    if ext_path:
        ext_path = os.path.abspath(ext_path)
        sys.path.insert(0, ext_path)
        if hasattr(os, "add_dll_directory") and os.path.isdir(ext_path):
            os.add_dll_directory(ext_path)
    if hasattr(os, "add_dll_directory"):
        for env_name in ("CUDA_PATH", "CUDA_PATH_V12_9", "CUDA_HOME"):
            cuda_root = os.environ.get(env_name)
            if not cuda_root:
                continue
            cuda_bin = os.path.join(cuda_root, "bin")
            if os.path.isdir(cuda_bin):
                os.add_dll_directory(cuda_bin)
    package_dir = Path(__file__).resolve().parent
    bundled_mode = os.environ.get("NSOS_MAMBA_RUNTIME", "auto").strip().lower()
    bundled_candidates: List[Path] = []
    if bundled_mode in ("auto", "cuda", "gpu"):
        bundled_candidates.extend(sorted(package_dir.glob("nsos_ext_cuda*.pyd")))
    if bundled_mode in ("auto", "cpu"):
        bundled_candidates.extend(sorted(package_dir.glob("nsos_ext_cpu*.pyd")))
    for candidate in bundled_candidates:
        try:
            return _load_extension_from_path(candidate)
        except Exception:
            sys.modules.pop("nsos_ext", None)
            if bundled_mode not in ("auto", ""):
                raise
    try:
        import nsos_ext as nsos  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "nsos_ext is not importable. Add the built nsos_ext directory to "
            "PYTHONPATH/sys.path before importing nsos_mamba."
        ) from exc
    return nsos


def _torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def detect_device(nsos_ext=None, requested: Optional[str] = None):
    """Resolve CPU/GPU from a flag or ``NSOS_MAMBA_DEVICE``.

    Accepted values: ``auto`` (default), ``gpu``/``cuda`` and ``cpu``.
    ``auto`` selects GPU only when PyTorch can see CUDA; construction still has
    a CPU fallback in ``NSOSMamba`` in case the native extension cannot allocate
    on the selected device.
    """

    nsos = nsos_ext or _load_nsos_ext()
    mode = (requested or os.environ.get("NSOS_MAMBA_DEVICE", "auto")).strip().lower()
    if mode in ("cpu", "host"):
        return nsos.Device.CPU
    if mode in ("gpu", "cuda"):
        return nsos.Device.GPU
    if mode not in ("", "auto"):
        raise ValueError("device must be one of: auto, gpu, cuda, cpu")
    return nsos.Device.GPU if _torch_cuda_available() else nsos.Device.CPU


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def _safe_set(obj, name: str, value) -> None:
    try:
        setattr(obj, name, value)
    except Exception:
        pass


@dataclass
class MambaModuleConfig:
    """Configuration for the standalone faithful-Mamba wrapper."""

    vocab_size: int
    num_layers: int = 12
    d_model: int = 128
    d_state: int = 64
    max_context_tokens: int = 4096
    n_heads: int = 4
    n_kv_heads: int = 2
    mamba_expand: int = 2
    mamba_head_dim: int = 64
    mamba_n_groups: int = 1
    use_attention: bool = False
    attention_period: int = 2
    attention_slot: int = 1
    tie_word_embeddings: bool = False
    ternary: bool = False
    ternary_warmup_steps: int = 100
    ternary_start_step: int = 300
    ternary_regularization: float = 1e-3
    seed: Optional[int] = 42
    device: str = "auto"
    streaming: bool = True
    allow_cpu_fallback: bool = True

    @classmethod
    def from_env(cls, vocab_size: int) -> "MambaModuleConfig":
        return cls(
            vocab_size=vocab_size,
            num_layers=int(os.environ.get("NSOS_MAMBA_LAYERS", "12")),
            d_model=int(os.environ.get("NSOS_MAMBA_DMODEL", "128")),
            d_state=int(os.environ.get("NSOS_MAMBA_DSTATE", "64")),
            max_context_tokens=int(os.environ.get("NSOS_MAMBA_CONTEXT", "4096")),
            use_attention=_env_bool("NSOS_MAMBA_ATTENTION", False),
            attention_period=int(os.environ.get("NSOS_MAMBA_ATTENTION_PERIOD", "2")),
            attention_slot=int(os.environ.get("NSOS_MAMBA_ATTENTION_SLOT", "1")),
            ternary=_env_bool("NSOS_MAMBA_TERNARY", False),
            ternary_warmup_steps=int(os.environ.get("NSOS_MAMBA_TERNARY_WARMUP", "100")),
            ternary_start_step=int(os.environ.get("NSOS_MAMBA_TERNARY_START", "300")),
            ternary_regularization=float(os.environ.get("NSOS_MAMBA_TERNARY_REG", "1e-3")),
            device=os.environ.get("NSOS_MAMBA_DEVICE", "auto"),
            streaming=_env_bool("NSOS_MAMBA_STREAMING", True),
        )


@dataclass
class GenerationResult:
    prompt_ids: List[int]
    generated_ids: List[int]
    text: Optional[str]
    prefill_ms: float
    decode_ms: float
    ttft_ms: float
    first_incremental_ms: float
    generated_tokens: int
    decode_tokens_per_sec: float
    end_to_end_tokens_per_sec: float
    streaming: bool


@dataclass
class DecodeBenchmark:
    runs: List[GenerationResult] = field(default_factory=list)

    @property
    def avg_decode_tokens_per_sec(self) -> float:
        return float(np.mean([r.decode_tokens_per_sec for r in self.runs])) if self.runs else 0.0

    @property
    def avg_end_to_end_tokens_per_sec(self) -> float:
        return float(np.mean([r.end_to_end_tokens_per_sec for r in self.runs])) if self.runs else 0.0

    @property
    def avg_prefill_ms(self) -> float:
        return float(np.mean([r.prefill_ms for r in self.runs])) if self.runs else 0.0

    @property
    def avg_first_incremental_ms(self) -> float:
        return float(np.mean([r.first_incremental_ms for r in self.runs])) if self.runs else 0.0

    def print(self) -> None:
        print("\n========== NSOS MAMBA DECODE BENCH ==========")
        for idx, run in enumerate(self.runs):
            print(
                f"run {idx}: {run.generated_tokens} tok | "
                f"decode={run.decode_ms / 1000.0:.4f}s | "
                f"decode_tok/s={run.decode_tokens_per_sec:.2f} | "
                f"e2e_tok/s={run.end_to_end_tokens_per_sec:.2f} | "
                f"prefill={run.prefill_ms:.2f} ms | "
                f"first_incremental={run.first_incremental_ms:.2f} ms"
            )
        print("---------------------------------------------")
        print("avg decode tok/s:", self.avg_decode_tokens_per_sec)
        print("avg e2e tok/s:", self.avg_end_to_end_tokens_per_sec)
        print("avg prefill ms:", self.avg_prefill_ms)
        print("avg first incremental ms:", self.avg_first_incremental_ms)


class NSOSMamba:
    """Pure faithful-Mamba NSOS module with streaming incremental by default."""

    def __init__(self, config: MambaModuleConfig, nsos_ext=None):
        self.nsos = nsos_ext or _load_nsos_ext()
        self.config = config
        if config.seed is not None:
            self.nsos.set_seed(int(config.seed))

        requested_device = detect_device(self.nsos, config.device)
        self.device = requested_device
        try:
            self.model = self.nsos.JambaModel(self._make_native_config(), requested_device)
            self.model.to(requested_device)
        except Exception:
            if not config.allow_cpu_fallback or requested_device == self.nsos.Device.CPU:
                raise
            self.device = self.nsos.Device.CPU
            self.model = self.nsos.JambaModel(self._make_native_config(), self.device)
            self.model.to(self.device)

        self.set_streaming(config.streaming)

    def _make_native_config(self):
        cfg = self.nsos.ModelConfig()
        cfg.num_layers = int(self.config.num_layers)
        cfg.d_model = int(self.config.d_model)
        cfg.vocab_size = int(self.config.vocab_size)
        cfg.n_heads = int(self.config.n_heads)
        cfg.n_kv_heads = int(self.config.n_kv_heads)
        cfg.max_context_tokens = int(self.config.max_context_tokens)

        cfg.use_moe = False
        cfg.use_kan = False
        cfg.use_ttt = False
        cfg.dropout = 0.0

        cfg.mamba2_faithful = True
        cfg.mamba_expand = int(self.config.mamba_expand)
        cfg.mamba_head_dim = int(self.config.mamba_head_dim)
        cfg.tie_word_embeddings = bool(self.config.tie_word_embeddings)

        _safe_set(cfg, "mamba_state_expansion", int(self.config.d_state))
        _safe_set(cfg, "mamba_n_groups", int(self.config.mamba_n_groups))

        if self.config.use_attention:
            cfg.attention_period = max(1, int(self.config.attention_period))
            cfg.attention_slot = max(0, int(self.config.attention_slot))
        else:
            # Disable attention by putting the slot outside any real layer schedule.
            cfg.attention_period = 1_000_000
            cfg.attention_slot = 0
        return cfg

    @property
    def vocab_size(self) -> int:
        return int(self.config.vocab_size)

    def set_streaming(self, enabled: bool = True) -> bool:
        try:
            if enabled and not self.model.supports_streaming_inference():
                self.model.set_streaming_inference(False)
                return False
            self.model.set_streaming_inference(bool(enabled))
            return bool(enabled)
        except Exception:
            return False

    def _logits_numpy(self, tensor) -> np.ndarray:
        return np.asarray(tensor.cpu().numpy()).reshape(-1, self.vocab_size)

    def fit_token_pairs(
        self,
        pairs: Sequence[Tuple[Sequence[int], Sequence[int]]],
        *,
        steps: int = 1200,
        batch_size: int = 16,
        learning_rate: float = 3e-3,
        qat: Optional[bool] = None,
        seed: int = 123,
        progress_every: Optional[int] = None,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> List[float]:
        if not pairs:
            raise ValueError("fit_token_pairs requires at least one pair")

        trainer = self.nsos.Trainer(self.model, float(learning_rate))
        trainer.warmup_steps = max(1, min(50, steps // 10 if steps > 10 else 1))
        trainer.total_training_steps = int(steps)
        trainer.first_token_loss_scale = 1.0
        trainer.eos_loss_scale = 1.0
        use_qat = self.config.ternary if qat is None else bool(qat)
        trainer.phase_scheduler.progressive_qat_enabled = bool(use_qat)
        trainer.phase_scheduler.semantic_warmup_steps = int(self.config.ternary_warmup_steps)
        trainer.phase_scheduler.qat_start_step = int(self.config.ternary_start_step)
        trainer.phase_scheduler.ternary_regularization = float(self.config.ternary_regularization)

        rng = random.Random(seed)
        self.model.set_training_mode(True)
        self.set_streaming(False)

        losses: List[float] = []
        window = progress_every or max(1, steps // 6)
        running = 0.0
        for step in range(int(steps)):
            prompts: List[List[int]] = []
            answers: List[List[int]] = []
            for _ in range(int(batch_size)):
                prompt, answer = rng.choice(pairs)
                prompts.append(list(prompt))
                answers.append(list(answer))
            loss = float(trainer.train_supervised_batch(prompts, answers))
            losses.append(loss)
            running += loss
            if callback and (step + 1) % window == 0:
                callback(step + 1, running / float(window))
                running = 0.0

        self.model.set_training_mode(False)
        self.set_streaming(self.config.streaming)
        return losses

    def fit_text_pairs(
        self,
        pairs: Sequence[TextPair],
        tokenizer: CharTokenizer,
        **kwargs,
    ) -> List[float]:
        token_pairs = pairs_to_token_ids(pairs, tokenizer)
        return self.fit_token_pairs(token_pairs, **kwargs)

    def generate_ids(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int = 128,
        eos_token_id: Optional[int] = None,
        stop_on_eos: bool = True,
        streaming: Optional[bool] = None,
        tokenizer: Optional[CharTokenizer] = None,
    ) -> GenerationResult:
        if max_new_tokens <= 0:
            return GenerationResult(
                prompt_ids=list(prompt_ids),
                generated_ids=[],
                text="",
                prefill_ms=0.0,
                decode_ms=0.0,
                ttft_ms=0.0,
                first_incremental_ms=0.0,
                generated_tokens=0,
                decode_tokens_per_sec=0.0,
                end_to_end_tokens_per_sec=0.0,
                streaming=bool(streaming),
            )

        use_streaming = self.config.streaming if streaming is None else bool(streaming)
        self.model.set_training_mode(False)
        self.set_streaming(use_streaming)

        if use_streaming:
            return self._generate_ids_streaming(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                stop_on_eos=stop_on_eos,
                tokenizer=tokenizer,
            )
        return self._generate_ids_eager(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            stop_on_eos=stop_on_eos,
            tokenizer=tokenizer,
        )

    def _generate_ids_streaming(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int,
        eos_token_id: Optional[int],
        stop_on_eos: bool,
        tokenizer: Optional[CharTokenizer],
    ) -> GenerationResult:
        self.model.reset_session()
        prompt = list(prompt_ids)

        prefill_t0 = time.perf_counter()
        logits = self._logits_numpy(self.model.forward_ids(prompt))
        prefill_s = time.perf_counter() - prefill_t0

        next_id = int(np.argmax(logits[-1]))
        generated = [next_id]
        if stop_on_eos and eos_token_id is not None and next_id == eos_token_id:
            text = tokenizer.decode(generated) if tokenizer else None
            return self._result(prompt, generated, text, prefill_s, 0.0, 0.0, True)

        decode_t0 = time.perf_counter()
        first_incremental_s = 0.0
        for idx in range(1, max_new_tokens):
            step_t0 = time.perf_counter()
            logits = self._logits_numpy(self.model.forward_ids([next_id]))
            if idx == 1:
                first_incremental_s = time.perf_counter() - step_t0
            next_id = int(np.argmax(logits[-1]))
            generated.append(next_id)
            if stop_on_eos and eos_token_id is not None and next_id == eos_token_id:
                break
        decode_s = time.perf_counter() - decode_t0
        text = tokenizer.decode(generated) if tokenizer else None
        return self._result(prompt, generated, text, prefill_s, decode_s, first_incremental_s, True)

    def _generate_ids_eager(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int,
        eos_token_id: Optional[int],
        stop_on_eos: bool,
        tokenizer: Optional[CharTokenizer],
    ) -> GenerationResult:
        prompt = list(prompt_ids)
        seq = list(prompt)
        generated: List[int] = []
        t0 = time.perf_counter()
        first_s = 0.0
        for idx in range(max_new_tokens):
            step_t0 = time.perf_counter()
            logits = self._logits_numpy(self.model.forward_ids(seq))
            if idx == 0:
                first_s = time.perf_counter() - step_t0
            next_id = int(np.argmax(logits[-1]))
            generated.append(next_id)
            seq.append(next_id)
            if stop_on_eos and eos_token_id is not None and next_id == eos_token_id:
                break
        total_s = time.perf_counter() - t0
        text = tokenizer.decode(generated) if tokenizer else None
        return self._result(prompt, generated, text, first_s, max(total_s - first_s, 0.0), first_s, False)

    def _result(
        self,
        prompt: List[int],
        generated: List[int],
        text: Optional[str],
        prefill_s: float,
        decode_s: float,
        first_incremental_s: float,
        streaming: bool,
    ) -> GenerationResult:
        # In streaming mode the first generated token comes from prefill; steady
        # decode throughput counts only tokens produced after prefill.
        steady_tokens = max(len(generated) - 1, 0) if streaming else len(generated)
        decode_tps = steady_tokens / max(decode_s, 1e-9)
        e2e_tps = len(generated) / max(prefill_s + decode_s, 1e-9)
        return GenerationResult(
            prompt_ids=prompt,
            generated_ids=generated,
            text=text,
            prefill_ms=prefill_s * 1000.0,
            decode_ms=decode_s * 1000.0,
            ttft_ms=prefill_s * 1000.0,
            first_incremental_ms=first_incremental_s * 1000.0,
            generated_tokens=len(generated),
            decode_tokens_per_sec=decode_tps,
            end_to_end_tokens_per_sec=e2e_tps,
            streaming=streaming,
        )

    def generate_text(
        self,
        prompt: str,
        tokenizer: CharTokenizer,
        *,
        max_new_tokens: int = 128,
        stop_on_eos: bool = True,
    ) -> GenerationResult:
        prompt_ids = tokenizer.encode(prompt, bos=True, eos=False)
        return self.generate_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_id,
            stop_on_eos=stop_on_eos,
            tokenizer=tokenizer,
        )

    def benchmark_decode(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int = 256,
        repeats: int = 5,
        warmup_tokens: int = 8,
        eos_token_id: Optional[int] = None,
    ) -> DecodeBenchmark:
        # Warmup: one short streaming run, no EOS stop.
        self.generate_ids(
            prompt_ids,
            max_new_tokens=max(1, warmup_tokens),
            eos_token_id=eos_token_id,
            stop_on_eos=False,
            streaming=True,
        )
        runs = [
            self.generate_ids(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                stop_on_eos=False,
                streaming=True,
            )
            for _ in range(repeats)
        ]
        return DecodeBenchmark(runs)
