"""NsosAdapter — wraps a loaded NSOS pack so the eval suite can score it.

This is the production adapter.  Loads a `.bin` model pack via
`nsos_ext`, exposes the uniform `ModelAdapter` interface so benchmarks
need not know about NSOS internals.

Token scoring strategy:
  We call `model.forward_ids(prompt + target)` to get logits over the
  whole sequence.  The logprob of `target_ids[i]` is the i-th element
  of softmax over `logits[len(prompt) + i - 1]`.  We then sum these.

Generation:
  We use `InferenceEngine.generate` when available (it includes top-k
  / temperature inside the C++ runtime).  Fallback to a Python loop
  over forward_ids + sampling if the engine isn't constructed.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import List, Optional

from .base import ModelAdapter, AdapterCapability


class NsosAdapter(ModelAdapter):
    """Wraps an NSOS pack loaded via nsos_ext."""

    def __init__(self, pack_path: str | Path, build_dir: str | Path,
                 *, device: str = "auto", tokenizer_path: Optional[str | Path] = None,
                 max_seq_len: int = 4096, eos_token: str = "<|endoftext|>"):
        """Load an NSOS pack and prepare it for scoring.

        Args:
          pack_path: path to the .bin checkpoint produced by NSOS Trainer.
          build_dir: NSOS build directory containing nsos_ext.{pyd,so}.
          device: 'auto' / 'gpu' / 'cpu'.  'auto' picks GPU if available.
          tokenizer_path: optional explicit tokenizer.  If None, expects
            `<run_dir>/tokenizer.nsos` next to the pack — that's how
            train_curriculum.py emits them.
          max_seq_len: hard cap for prompt+target length.
          eos_token: literal text used to end generation.

        Lazy imports nsos_ext only when an NsosAdapter is constructed,
        so importing this module from a build-less environment is fine.
        """
        self.pack_path = Path(pack_path)
        self.build_dir = Path(build_dir)
        self._max_seq_len = max_seq_len
        self._eos_token = eos_token
        self._device_request = device

        nsos = self._import_nsos()
        self._nsos = nsos

        # Locate tokenizer
        tok = nsos.Tokenizer()
        if tokenizer_path is None:
            cand = self.pack_path.with_name("tokenizer.nsos")
            if not cand.exists():
                cand = self.pack_path.parent / "tokenizer.nsos"
            if not cand.exists():
                raise FileNotFoundError(
                    f"No tokenizer.nsos found near {self.pack_path}.  "
                    f"Pass tokenizer_path explicitly."
                )
            tokenizer_path = cand
        try:
            tok.load_pack(str(tokenizer_path))
        except Exception:
            tok.load(str(tokenizer_path))
        self.tokenizer = tok
        self._vocab_size = int(tok.vocab_size)

        # Resolve device
        if device == "auto":
            self.device = nsos.Device.GPU if nsos.fast_gpu_supported() else nsos.Device.CPU
        elif device == "gpu":
            self.device = nsos.Device.GPU
        else:
            self.device = nsos.Device.CPU

        # We need the model_config to construct JambaModel.  This requires
        # an `effective_model_config.json` alongside the pack — that's
        # what train_curriculum.py writes when it saves checkpoints.
        import json
        cfg_path = self.pack_path.with_name("effective_model_config.json")
        if not cfg_path.exists():
            cfg_path = self.pack_path.parent / "effective_model_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"Need effective_model_config.json next to the pack to "
                f"reconstruct the model.  Looked at {self.pack_path.parent}/."
            )
        cfg_data = json.loads(Path(cfg_path).read_text(encoding="utf-8"))

        mc = nsos.ModelConfig()
        # Apply every supported field — silently ignore unknown ones so
        # we don't break on schema additions.
        for key, value in cfg_data.items():
            if hasattr(mc, key):
                try:
                    setattr(mc, key, value)
                except (TypeError, ValueError):
                    pass

        self.model = nsos.JambaModel(mc, self.device)
        self.model.to(self.device)
        # strict=False so we tolerate minor schema drift between checkpoint
        # and current binary; benchmarks will surface real quality drops if
        # something's mismatched, more useful than a hard load failure.
        try:
            self.model.load(str(self.pack_path), True)
            self._load_mode = "strict"
        except RuntimeError:
            self.model.load(str(self.pack_path), False)
            self._load_mode = "partial"
        self.model.set_training_mode(False)

        # Param count: best-effort from parameters() if exposed
        n_params = 0
        try:
            params = self.model.parameters()
            for p in params:
                # parameters() may be a vector of Parameter objects with .data
                data = getattr(p, "data", p)
                try:
                    n_params += int(data.size)
                except AttributeError:
                    n_params += 0
        except Exception:
            n_params = 0
        self._param_count = n_params

        # Cached EOS token id for generation stop signal
        try:
            self._eos_id = int(self.tokenizer.encode(eos_token)[0])
        except Exception:
            self._eos_id = -1

    # ─── ModelAdapter interface ────────────────────────────────────────

    @property
    def capability(self) -> AdapterCapability:
        return AdapterCapability(
            can_score_tokens=True,
            can_generate=True,
            can_batch_score=False,  # NSOS bindings don't yet expose batched logits cleanly
            max_seq_len=self._max_seq_len,
            vocab_size=self._vocab_size,
            model_name=f"nsos:{self.pack_path.name}",
            model_params=self._param_count,
            notes=(f"NSOS pack loaded {self._load_mode}-ly on "
                   f"{'GPU' if self.device == self._nsos.Device.GPU else 'CPU'}"),
        )

    def tokenize(self, text: str) -> List[int]:
        return list(self.tokenizer.encode(text))

    def detokenize(self, ids: List[int]) -> str:
        return self.tokenizer.decode(list(ids))

    def score_tokens(self, prompt_ids: List[int], target_ids: List[int]) -> float:
        """sum_i log softmax(forward(prompt + target_<i))[target_i]"""
        if not target_ids:
            return 0.0
        full = list(prompt_ids) + list(target_ids)
        target_ids = list(target_ids)
        if len(full) > self._max_seq_len:
            # Truncate from the LEFT so target_ids stay intact — they're
            # what we score.  Drop oldest prompt tokens.
            drop = len(full) - self._max_seq_len
            full = full[drop:]
            # Recompute where the target starts within the truncated view.
            # May go NEGATIVE when truncation ate past the prompt into the
            # target itself — the unclamped value tells us how many target
            # tokens were dropped from `full`.
            prompt_len = len(prompt_ids) - drop
        else:
            prompt_len = len(prompt_ids)
        if prompt_len < 1:
            # Prompt fully truncated: leading target tokens either fell
            # out of `full` entirely (prompt_len negative) or have no
            # predecessor row to score from (prompt_len == 0).  Treat
            # them as unscored context (lm-eval convention) — otherwise
            # arr[prompt_len-1] would be arr[-1] and numpy wraps to the
            # LAST row.
            skip = 1 - prompt_len
            target_ids = target_ids[skip:]
            prompt_len = 1
            if not target_ids:
                return 0.0

        # Run forward and pull logits as numpy.  forward_ids returns a Tensor
        # of shape (seq_len, vocab_size); the i-th row predicts token i+1.
        logits = self.model.forward_ids(full)
        arr = logits.cpu().numpy()  # (seq_len, vocab_size)
        # Some bindings return (1, seq_len, vocab) or (seq_len, vocab).
        if arr.ndim == 3:
            arr = arr[0]
        # logprob of target_ids[i] = log_softmax(arr[prompt_len + i - 1])[target_ids[i]]
        import numpy as np
        total_logprob = 0.0
        for i, t in enumerate(target_ids):
            row = arr[prompt_len + i - 1]
            # Stable log-softmax
            m = float(np.max(row))
            logsumexp = m + math.log(float(np.sum(np.exp(row - m))))
            total_logprob += float(row[t]) - logsumexp
        return total_logprob

    def generate(self, prompt_text: str, *, max_new_tokens: int = 256,
                  temperature: float = 0.7, top_k: int = 40,
                  stop_sequences: Optional[List[str]] = None) -> str:
        """Token-by-token sampling on the GPU/CPU path.

        We DO NOT use InferenceEngine.generate here because constructing
        an engine requires the full SDK pipeline (pack load, etc.); we
        already have a JambaModel ready.  We do our own sampling loop on
        top of forward_ids — slower but always available.
        """
        import zlib

        import numpy as np

        ids = self.tokenize(prompt_text)
        generated_ids: List[int] = []
        eos_seen = False

        # zlib.crc32 is stable across processes; Python's hash() is
        # randomized per run (PYTHONHASHSEED) and would make eval
        # generations irreproducible.
        rng = np.random.default_rng(seed=zlib.crc32(prompt_text.encode("utf-8")))

        for _ in range(max_new_tokens):
            ctx = ids + generated_ids
            if len(ctx) > self._max_seq_len:
                ctx = ctx[-self._max_seq_len:]
            logits = self.model.forward_ids(ctx)
            arr = logits.cpu().numpy()
            if arr.ndim == 3:
                arr = arr[0]
            last_row = arr[-1].astype(np.float64)

            if temperature <= 1e-6:
                # Greedy
                next_id = int(np.argmax(last_row))
            else:
                # Top-k + temperature sampling
                row = last_row / max(temperature, 1e-6)
                if top_k > 0 and top_k < self._vocab_size:
                    cutoff = np.partition(row, -top_k)[-top_k]
                    row = np.where(row >= cutoff, row, -np.inf)
                m = float(np.max(row))
                probs = np.exp(row - m)
                probs = probs / probs.sum()
                next_id = int(rng.choice(self._vocab_size, p=probs))

            generated_ids.append(next_id)
            if next_id == self._eos_id:
                eos_seen = True
                break

            if stop_sequences:
                current = self.detokenize(generated_ids)
                if any(s in current for s in stop_sequences):
                    # Trim back so the stop sequence isn't in output
                    for s in stop_sequences:
                        idx = current.find(s)
                        if idx >= 0:
                            current = current[:idx]
                            break
                    return current

        if eos_seen and generated_ids and generated_ids[-1] == self._eos_id:
            generated_ids = generated_ids[:-1]
        return self.detokenize(generated_ids)

    # ─── helpers ───────────────────────────────────────────────────────

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
