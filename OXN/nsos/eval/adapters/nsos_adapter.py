"""Evaluate the exact model and tokenizer loaded by the serving SDK."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Optional

from .base import ModelAdapter, AdapterCapability


class NsosAdapter(ModelAdapter):
    def __init__(self, pack_path: str | Path, build_dir: str | Path,
                 *, device: str = "auto", tokenizer_path: Optional[str | Path] = None,
                 max_seq_len: int = 4096, eos_token: str = "<|endoftext|>"):
        self.pack_path = Path(pack_path)
        self.build_dir = Path(build_dir)
        if device not in {"auto", "cpu", "gpu"} or max_seq_len < 2:
            raise ValueError("Invalid evaluation device or context limit")
        nsos = self._import_nsos()
        self._nsos = nsos
        gpu = device == "gpu" or (device == "auto" and nsos.fast_gpu_supported())
        self.device = nsos.Device.GPU if gpu else nsos.Device.CPU
        self.engine = nsos.InferenceEngine()

        identity_paths = {}
        if self.pack_path.is_dir():
            if tokenizer_path is not None:
                raise ValueError("Model-pack evaluation uses its checksummed tokenizer; overrides are forbidden")
            options = nsos.ModelLoadOptions()
            options.use_cuda = gpu
            loaded = self.engine.load_model(str(self.pack_path), options)
            identity_file = self.pack_path / "manifest.nsos"
            identity_paths["pack_manifest"] = identity_file
        else:
            cfg_path = self.pack_path.with_name("effective_model_config.json")
            cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
            if not isinstance(cfg_data, dict):
                raise ValueError("Model configuration must be an object")
            config = nsos.ModelConfig()
            for key, value in cfg_data.items():
                if not hasattr(config, key):
                    raise ValueError(f"Unsupported model configuration field: {key}")
                if key == "hybrid_composition":
                    value = nsos.HybridComposition(value)
                try:
                    setattr(config, key, value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid model configuration field: {key}") from exc
            config.use_cuda = gpu
            identity_paths["config"] = cfg_path
            loaded = self.engine.load_model(str(self.pack_path), config)
            if loaded:
                tokenizer = Path(tokenizer_path) if tokenizer_path else self.pack_path.with_name("tokenizer.nsos")
                self.engine.load_tokenizer(str(tokenizer))
                identity_paths["tokenizer"] = tokenizer
            identity_file = self.pack_path
            identity_paths["weights"] = identity_file

        if not loaded:
            raise RuntimeError(f"Strict model load failed: {self.pack_path}")
        config = self.engine.model_config()
        self._max_seq_len = min(max_seq_len, int(config.max_context_tokens))
        self._vocab_size = int(config.vocab_size)
        self._param_count = int(self.engine.parameter_count())
        digest = hashlib.sha256()
        with identity_file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        self._identity = digest.hexdigest()
        identity_paths["binary"] = Path(nsos.__file__)
        self._evaluation_identity = {
            "kind": "model", "training_status": "not_attested",
            "artifact_sha256": {name: self._file_sha256(path)
                                for name, path in identity_paths.items()},
            "device": str(self.device), "max_seq_len": self._max_seq_len,
            "vocab_size": self._vocab_size, "parameters": self._param_count,
            "runtime_environment": {key: value for key, value in sorted(os.environ.items())
                                    if key.startswith(("NSOS_GPU_", "NSOS_HIP_", "NSOS_MATMUL_"))},
        }
        eos_ids = self.engine.tokenize(eos_token)
        self._eos_id = int(eos_ids[0]) if len(eos_ids) == 1 else -1

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def evaluation_identity(self) -> dict:
        return self._evaluation_identity

    @property
    def capability(self) -> AdapterCapability:
        return AdapterCapability(
            can_score_tokens=True, can_generate=True, can_batch_score=False,
            max_seq_len=self._max_seq_len, vocab_size=self._vocab_size,
            model_name=f"nsos:{self.pack_path.name}",
            model_params=self._param_count,
            notes=f"SDK strict load; sha256={self._identity}; device={self.device}",
        )

    def tokenize(self, text: str) -> List[int]:
        return list(self.engine.tokenize(text))

    def detokenize(self, ids: List[int]) -> str:
        return self.engine.detokenize(list(ids))

    def score_tokens(self, prompt_ids: List[int], target_ids: List[int]) -> float:
        if not target_ids:
            return 0.0
        full = list(prompt_ids) + list(target_ids)
        targets = list(target_ids)
        prompt_len = len(prompt_ids)
        if prompt_len < 1 or len(targets) >= self._max_seq_len:
            raise ValueError("Token scoring requires context and room for every target")
        if len(full) > self._max_seq_len:
            drop = len(full) - self._max_seq_len
            full = full[drop:]
            prompt_len -= drop
        if prompt_len < 1:
            raise ValueError("Evaluation must not silently drop target tokens")
        import numpy as np
        logits = self.engine.forward_logits(full)
        arr = logits.cpu().numpy()
        if arr.ndim == 3:
            arr = arr[0]
        if arr.shape != (len(full), self._vocab_size) or not np.isfinite(arr).all():
            raise RuntimeError("Evaluation received invalid or non-finite logits")
        rows = arr[prompt_len - 1:prompt_len - 1 + len(targets)].astype(np.float64)
        maxima = rows.max(axis=1)
        normalizers = maxima + np.log(np.exp(rows - maxima[:, None]).sum(axis=1))
        total = float((rows[np.arange(len(targets)), targets] - normalizers).sum())
        if not math.isfinite(total):
            raise RuntimeError("Non-finite evaluation log probability")
        return total

    def generate(self, prompt_text: str, *, max_new_tokens: int = 256,
                 temperature: float = 0.7, top_k: int = 40,
                 stop_sequences: Optional[List[str]] = None) -> str:
        options = self._nsos.GenerationOptions()
        options.max_tokens = max_new_tokens
        options.temperature = temperature
        options.top_k = top_k
        options.max_context_tokens = self._max_seq_len
        if self._eos_id >= 0:
            options.eos_token_id = self._eos_id
        result = self.engine.generate_ex(prompt_text, options)
        if stop_sequences:
            positions = [result.find(stop) for stop in stop_sequences if stop]
            positions = [position for position in positions if position >= 0]
            if positions:
                result = result[:min(positions)]
        return result

    def _import_nsos(self):
        """Lazy import of nsos_ext with build-dir + CUDA on PATH for Windows."""
        build_dir = self.build_dir.resolve()
        if str(build_dir) not in sys.path:
            sys.path.insert(0, str(build_dir))

        if os.name == "nt":
            # On Windows, CUDA dlls must be discoverable.  Use the existing
            # NSOS helper if available.
            try:
                from cuda_env import add_windows_runtime_dirs, parse_preferred_cuda_root
                add_windows_runtime_dirs(
                    build_dir,
                    parse_preferred_cuda_root(os.environ.get("NSOS_CUDA_ROOT")),
                )
            except ImportError:
                # Try the OXN scripts dir
                scripts = Path(__file__).resolve().parents[2] / "scripts"
                if scripts.exists() and str(scripts) not in sys.path:
                    sys.path.insert(0, str(scripts))
                    from cuda_env import add_windows_runtime_dirs, parse_preferred_cuda_root
                    add_windows_runtime_dirs(
                        build_dir,
                        parse_preferred_cuda_root(os.environ.get("NSOS_CUDA_ROOT")),
                    )

        import nsos_ext  # type: ignore
        return nsos_ext
