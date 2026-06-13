from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, TextIO

import numpy as np

from cuda_env import add_windows_runtime_dirs, parse_preferred_cuda_root
from nsos_curriculum_lib import (
    PHASE_ORDER,
    PHASE_ORDER_V11,
    SPECIAL_TOKENS,
    build_curriculum,
    build_tokenizer_bundle,
    curriculum_texts_for_phase,
    resolve_phase_order,
)
from summarize_layer_audit import build_report as build_layer_audit_report

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


EVAL_RUNTIME_OPTIONS: Dict[str, int] = {
    "masked_batch_size": 4,
    "generation_probe_samples": 0,
    "fast_exact_samples": 0,
    "text_loss_max_windows": 8,
}


PROFILES: Dict[str, Dict] = {
    "smoke": {
        "layers": 2,
        "d_model": 96,
        "target_vocab": 1024,
        "batch_size": 2,
        "seq_len": 64,
        "lr": 0.0025,
        "weight_decay": 0.0,
        "max_grad_norm": 1.5,
        "warmup_steps": 6,
        "min_lr_scale": 0.25,
        "repetition_unlikelihood_scale": 0.0,
        "phase_repetition_unlikelihood_scale": {
            "phase1_algorithms": 0.0,
            "phase2_structured": 0.0,
            "phase3_curated_text": 0.0,
            "phase4_instructions": 0.0,
            "phase5_verifier": 0.0,
            "phase6_memory": 0.0,
            "instruction_polish": 0.0,
        },
        "instruction_polish_steps": 0,
        "instruction_polish_lr_scale": 1.0,
        "phase_steps": {
            "phase1_algorithms": 12,
            "phase2_structured": 12,
            "phase3_curated_text": 10,
            "phase4_instructions": 10,
            "phase5_verifier": 12,
            "phase6_memory": 10,
        },
        "qat": {
            "enabled": False,
            "semantic_warmup_steps": 0,
            "qat_start_step": 0,
            "quantized_precision_bits": 2,
            "ternary_regularization": 0.0,
        },
    },
    "pilot": {
        "layers": 4,
        "d_model": 128,
        "target_vocab": 2048,
        "batch_size": 3,
        "seq_len": 96,
        "lr": 0.0015,
        "weight_decay": 0.002,
        "max_grad_norm": 1.5,
        "warmup_steps": 10,
        "min_lr_scale": 0.2,
        "repetition_unlikelihood_scale": 0.03,
        "phase_repetition_unlikelihood_scale": {
            "phase1_algorithms": 0.0,
            "phase2_structured": 0.0,
            "phase3_curated_text": 0.0,
            "phase4_instructions": 0.04,
            "phase5_verifier": 0.015,
            "phase6_memory": 0.015,
            "instruction_polish": 0.05,
        },
        "instruction_polish_steps": 8,
        "instruction_polish_lr_scale": 0.55,
        "phase_steps": {
            "phase1_algorithms": 20,
            "phase2_structured": 20,
            "phase3_curated_text": 18,
            "phase4_instructions": 16,
            "phase5_verifier": 20,
            "phase6_memory": 16,
        },
        "qat": {
            # True BitNet b1.58 quantization-aware training: after an FP32
            # semantic warmup, the forward runs through the real packed ternary
            # kernel and back-propagates with a straight-through estimator onto
            # the FP32 latent weights (see src/bitlinear.cpp).  Use --disable_qat
            # to fall back to FP32 training + post-training quantization.
            "enabled": True,
            "semantic_warmup_steps": 12,
            "qat_start_step": 36,
            "quantized_precision_bits": 2,
            "ternary_regularization": 4e-4,
        },
    },
    "small": {
        "layers": 8,
        "d_model": 320,
        "target_vocab": 8192,
        "batch_size": 3,
        "seq_len": 160,
        "lr": 0.0005,
        "weight_decay": 0.003,
        "max_grad_norm": 0.9,
        "warmup_steps": 60,
        "min_lr_scale": 0.20,
        "first_token_loss_scale": 2.4,
        "eos_loss_scale": 0.55,
        "repetition_unlikelihood_scale": 0.06,
        "phase_repetition_unlikelihood_scale": {
            "phase1_algorithms": 0.0,
            "phase2_structured": 0.0,
            "phase3_curated_text": 0.0,
            "phase4_instructions": 0.07,
            "phase5_verifier": 0.02,
            "phase6_memory": 0.02,
            "instruction_polish": 0.09,
        },
        "instruction_polish_steps": 24,
        "instruction_polish_lr_scale": 0.45,
        "phase_steps": {
            "phase1_algorithms": 48,
            "phase2_structured": 32,
            "phase3_curated_text": 128,
            "phase4_instructions": 64,
            "phase5_verifier": 48,
            "phase6_memory": 72,
        },
        "phase_lr_scale": {
            "phase1_algorithms": 1.0,
            "phase2_structured": 0.95,
            "phase3_curated_text": 0.55,
            "phase4_instructions": 1.0,
            "phase5_verifier": 0.95,
            "phase6_memory": 0.90,
        },
        "qat": {
            # True BitNet b1.58 QAT (real packed ternary kernel + STE on FP32
            # latent weights).  Long FP32 warmup before quantizing.  Use
            # --disable_qat to fall back to FP32 + post-training quantization.
            "enabled": True,
            "semantic_warmup_steps": 160,
            "qat_start_step": 320,
            "quantized_precision_bits": 2,
            "ternary_regularization": 1.5e-4,
        },
    },
}

PROFILE_ALIASES = {
    "pilot": "hybrid_pilot",
    "small": "mamba_small",
}

PROFILES["mamba_small"] = deepcopy(PROFILES["small"])
PROFILES["mamba_small"]["profile_family"] = "mamba"
PROFILES["mamba_small"]["model_config"] = {
    "n_heads": 4,
    "n_kv_heads": 2,
    "sliding_window": 4096,
    "attention_period": 64,
    "attention_slot": 63,
    "use_moe": False,
    "num_experts": 4,
    "num_experts_per_token": 2,
    "moe_period": 64,
    "moe_slot": 63,
    "use_ttt": False,
    "ttt_period": 64,
    "ttt_slot": 63,
    "use_exact_attention_training": False,
    "use_flash_attn": False,
}
PROFILES["mamba_small"]["validation_scope"] = "Validates the production Mamba-first stack."

PROFILES["mamba_presentable"] = deepcopy(PROFILES["mamba_small"])
PROFILES["mamba_presentable"]["requested_role"] = "presentable_focus"
PROFILES["mamba_presentable"]["instruction_polish_steps"] = 8
PROFILES["mamba_presentable"]["instruction_polish_lr_scale"] = 0.30
PROFILES["mamba_presentable"]["phase_steps"] = {
    "phase1_algorithms": 80,
    "phase2_structured": 72,
    "phase3_curated_text": 96,
    "phase4_instructions": 32,
    "phase5_verifier": 24,
    "phase6_memory": 96,
}
PROFILES["mamba_presentable"]["phase_lr_scale"] = {
    "phase1_algorithms": 1.05,
    "phase2_structured": 1.00,
    "phase3_curated_text": 0.45,
    "phase4_instructions": 0.80,
    "phase5_verifier": 0.75,
    "phase6_memory": 1.00,
}
PROFILES["mamba_presentable"]["phase_sizes"] = {
    "phase1_algorithms": {"train": 1152, "eval": 192},
    "phase2_structured": {"train": 960, "eval": 160},
    "phase3_curated_text": {"train": 512, "eval": 96},
    "phase4_instructions": {"train": 256, "eval": 64},
    "phase5_verifier": {"train": 256, "eval": 64},
    "phase6_memory": {"train": 1152, "eval": 192},
}
PROFILES["mamba_presentable"]["qat"] = {
    "enabled": False,
    "semantic_warmup_steps": 0,
    "qat_start_step": 0,
    "quantized_precision_bits": 2,
    "ternary_regularization": 0.0,
}
PROFILES["mamba_presentable"]["validation_scope"] = (
    "Presentable Mamba-first profile that emphasizes algorithmic, structured, and memory behavior."
)

PROFILES["hybrid_pilot"] = deepcopy(PROFILES["pilot"])
PROFILES["hybrid_pilot"]["profile_family"] = "hybrid"
PROFILES["hybrid_pilot"]["model_config"] = {
    "n_heads": 4,
    "n_kv_heads": 2,
    "sliding_window": 2048,
    "attention_period": 2,
    "attention_slot": 2,
    "use_moe": False,
    "num_experts": 4,
    "num_experts_per_token": 2,
    "moe_period": 64,
    "moe_slot": 63,
    "use_ttt": False,
    "ttt_period": 64,
    "ttt_slot": 63,
    "use_exact_attention_training": True,
    "use_flash_attn": False,
}
PROFILES["hybrid_pilot"]["validation_scope"] = "Validates attention + Mamba without TTT."

PROFILES["hybrid_moe_smoke"] = deepcopy(PROFILES["pilot"])
PROFILES["hybrid_moe_smoke"]["profile_family"] = "hybrid"
PROFILES["hybrid_moe_smoke"]["requested_role"] = "moe_audit_smoke"
PROFILES["hybrid_moe_smoke"]["batch_size"] = 2
PROFILES["hybrid_moe_smoke"]["lr"] = 0.0012
PROFILES["hybrid_moe_smoke"]["warmup_steps"] = 12
PROFILES["hybrid_moe_smoke"]["instruction_polish_steps"] = 0
PROFILES["hybrid_moe_smoke"]["phase_steps"] = {
    "phase1_algorithms": 8,
    "phase2_structured": 8,
    "phase3_curated_text": 6,
    "phase4_instructions": 6,
    "phase5_verifier": 8,
    "phase6_memory": 6,
}
PROFILES["hybrid_moe_smoke"]["model_config"] = {
    "n_heads": 4,
    "n_kv_heads": 2,
    "sliding_window": 1024,
    "attention_period": 2,
    "attention_slot": 0,
    "use_moe": True,
    "num_experts": 4,
    "num_experts_per_token": 2,
    "moe_period": 2,
    "moe_slot": 1,
    "use_ttt": False,
    "ttt_period": 64,
    "ttt_slot": 63,
    "use_exact_attention_training": True,
    "use_flash_attn": False,
}
PROFILES["hybrid_moe_smoke"]["validation_scope"] = (
    "CPU-feasible MoE smoke lane for router coverage and quick holdout trend checks."
)

PROFILES["hybrid_moe_long"] = deepcopy(PROFILES["hybrid_moe_smoke"])
PROFILES["hybrid_moe_long"]["requested_role"] = "moe_capacity_probe"
PROFILES["hybrid_moe_long"]["instruction_polish_steps"] = 8
PROFILES["hybrid_moe_long"]["phase_steps"] = {
    "phase1_algorithms": 24,
    "phase2_structured": 24,
    "phase3_curated_text": 20,
    "phase4_instructions": 20,
    "phase5_verifier": 24,
    "phase6_memory": 20,
}
PROFILES["hybrid_moe_long"]["validation_scope"] = (
    "Longer compact-audit MoE lane for capacity checks after performance probes are green."
)

PROFILES["hybrid_moe_capacity"] = deepcopy(PROFILES["hybrid_moe_long"])
PROFILES["hybrid_moe_capacity"]["requested_role"] = "moe_capacity_holdout_champion"
PROFILES["hybrid_moe_capacity"]["instruction_polish_steps"] = 0
PROFILES["hybrid_moe_capacity"]["moe_aux_loss_scale"] = 0.50
PROFILES["hybrid_moe_capacity"]["kind_regression_penalty_scale"] = 0.85
PROFILES["hybrid_moe_capacity"]["kind_regression_floor"] = 0.20
PROFILES["hybrid_moe_capacity"]["kind_regression_gate"] = {
    "enabled": True,
    "max_regression": 2.25,
}
PROFILES["hybrid_moe_capacity"]["weak_kind_rehearsal_steps"] = 24
PROFILES["hybrid_moe_capacity"]["weak_kind_rehearsal_lr_scale"] = 0.45
PROFILES["hybrid_moe_capacity"]["weak_kind_minipack"] = {
    "enabled": True,
    "kinds": [
        "binary_add",
        "boolean_formula",
        "circuit",
        "code_output",
        "compare",
        "compare_label",
        "convert_json",
        "dsl",
        "kv_to_csv",
        "logparse",
        "memory_recall",
        "parity",
        "parity_label",
        "reverse",
        "summarize",
        "translate",
    ],
    "phase_counts": {
        "phase2_structured": 160,
        "phase4_instructions": 224,
        "phase5_verifier": 320,
        "phase6_memory": 384,
        "weak_kind_rehearsal": 768,
    },
}
PROFILES["hybrid_moe_capacity"]["phase_replay_config"] = {
    "phase2_structured": {"families": ["algorithmic"], "ratio_scale": 0.50},
    "phase4_instructions": {"families": ["algorithmic", "structured"], "ratio_scale": 0.75},
    "phase5_verifier": {
        "families": ["algorithmic", "structured", "instruction"],
        "ratio_scale": 1.50,
    },
    "phase6_memory": {
        "families": ["algorithmic", "structured", "instruction", "verifier"],
        "ratio_scale": 2.00,
    },
}
PROFILES["hybrid_moe_capacity"]["validation_scope"] = (
    "MoE capacity lane that preserves the best official-holdout checkpoint without the polish tail."
)

PROFILES["hybrid_small"] = deepcopy(PROFILES["small"])
PROFILES["hybrid_small"]["profile_family"] = "hybrid"
PROFILES["hybrid_small"]["model_config"] = {
    "n_heads": 8,
    "n_kv_heads": 4,
    "sliding_window": 4096,
    "attention_period": 2,
    "attention_slot": 2,
    "use_moe": True,
    "num_experts": 6,
    "num_experts_per_token": 2,
    "moe_period": 3,
    "moe_slot": 3,
    "use_ttt": False,
    "ttt_period": 64,
    "ttt_slot": 63,
    "use_exact_attention_training": True,
    "use_flash_attn": False,
}
PROFILES["hybrid_small"]["validation_scope"] = (
    "Validates attention + sparse MoE + Mamba; TTT remains research-only."
)

PROFILES["hybrid_medium"] = deepcopy(PROFILES["small"])
PROFILES["hybrid_medium"]["profile_family"] = "hybrid"
PROFILES["hybrid_medium"]["layers"] = 12
PROFILES["hybrid_medium"]["d_model"] = 512
PROFILES["hybrid_medium"]["lr"] = 4e-4
PROFILES["hybrid_medium"]["warmup_steps"] = 80
PROFILES["hybrid_medium"]["model_config"] = {
    "n_heads": 8,
    "n_kv_heads": 4,
    "sliding_window": 4096,
    "attention_period": 2,
    "attention_slot": 2,
    "use_moe": True,
    "num_experts": 8,
    "num_experts_per_token": 2,
    "moe_period": 3,
    "moe_slot": 3,
    "use_ttt": False,
    "ttt_period": 64,
    "ttt_slot": 63,
    "use_exact_attention_training": True,
    "use_flash_attn": False,
}
PROFILES["hybrid_medium"]["phase_steps"] = {
    "phase1_algorithms": 0,
    "phase2_structured": 0,
    "phase3_curated_text": 320,
    "phase4_instructions": 352,
    "phase5_verifier": 0,
    "phase6_memory": 0,
}
PROFILES["hybrid_medium"]["phase_lr_scale"] = {
    "phase1_algorithms": 1.0,
    "phase2_structured": 0.95,
    "phase3_curated_text": 0.55,
    "phase4_instructions": 1.0,
    "phase5_verifier": 1.0,
    "phase6_memory": 1.0,
}
PROFILES["hybrid_medium"]["instruction_polish_steps"] = 128
PROFILES["hybrid_medium"]["instruction_polish_lr_scale"] = 0.70
PROFILES["hybrid_medium"]["validation_scope"] = (
    "hybrid_medium: 12-layer d_model=512 hybrid with MoE-8. ~40M params. Train from scratch."
)

# ── hybrid_medium_v9 ──────────────────────────────────────────────────────
# Identical architecture to hybrid_medium. Key changes vs v8:
#   1. Use distillation_bundle_v4 which has a CLEAN English tokenizer rebuilt
#      from Wikipedia (v3 tokenizer was C++-contaminated — built before phase3
#      was replaced with Wikipedia data, so all generation defaulted to
#      ensure_kv_cache_capac as the single most likely token).
#   2. Added small phase1/phase2 heads (24+16 steps) — these tasks are pure
#      numeric/symbolic so they're fine with any tokenizer and give the model
#      algorithmic grounding before Wikipedia.
#   3. Increased phase3 Wikipedia steps: 320 → 480 (more English pretraining).
#   4. Increased instruction steps: 352 → 448 (model can now actually learn
#      English responses, so more instruction signal is beneficial).
#   5. Increased polish steps: 128 → 192.
#   6. warmup_steps increased: 80 → 100 (larger total budget).
# Total training: ~1164 steps vs ~800 in hybrid_medium.
PROFILES["hybrid_medium_v9"] = deepcopy(PROFILES["hybrid_medium"])
PROFILES["hybrid_medium_v9"]["warmup_steps"] = 100
PROFILES["hybrid_medium_v9"]["phase_steps"] = {
    "phase1_algorithms": 24,
    "phase2_structured": 16,
    "phase3_curated_text": 480,
    "phase4_instructions": 448,
    "phase5_verifier": 0,
    "phase6_memory": 0,
}
PROFILES["hybrid_medium_v9"]["phase_lr_scale"] = {
    "phase1_algorithms": 0.90,
    "phase2_structured": 0.85,
    "phase3_curated_text": 0.55,
    "phase4_instructions": 1.0,
    "phase5_verifier": 1.0,
    "phase6_memory": 1.0,
}
PROFILES["hybrid_medium_v9"]["instruction_polish_steps"] = 192
PROFILES["hybrid_medium_v9"]["instruction_polish_lr_scale"] = 0.65
PROFILES["hybrid_medium_v9"]["validation_scope"] = (
    "hybrid_medium_v9: Same arch as hybrid_medium. Uses distillation_bundle_v4 with "
    "English-first BPE tokenizer (8189 tokens learned from Wikipedia). "
    "Fixes v8 tokenizer contamination where C++ identifiers dominated the vocab. "
    "Larger phase3 Wikipedia (480 steps) and instruction (448 steps) budgets."
)

# ── hybrid_medium_v10 ─────────────────────────────────────────────────────
# v9 observations (training run on CPU step 0..370/480 in phase3):
#   * Loss dropped sharply 10.79 → 6.82 in first 100 steps then plateaued
#     between 6.85 and 7.40 for the next 270 steps with non-trivial
#     oscillation (batch=3 is too small — variance dominated the signal).
#   * Phase3 LR (2.20e-04) was probably too high once the easy mass of
#     the distribution was learned; the model kept stepping over local
#     minima rather than refining.
#   * Phase4 / instruction polish never ran because we killed v9 at
#     step 370 to free the GPU for testing.
#
# v10 changes vs v9:
#   1. Lower phase3 LR scale (0.55 → 0.40) — slower learning during the
#      Wikipedia plateau region.  Aims to coax the loss below 6.5 EMA.
#   2. Longer warmup (100 → 160) — gentler LR ramp reduces phase3 entry
#      shock that cost ~10 steps of oscillation in v9.
#   3. Phase4 instructions: lower LR scale (1.0 → 0.85) for stable
#      instruction adaptation given the already-trained phase3 features.
#   4. Polish steps stay at 192 with same lr_scale (0.65); we already
#      saw that's adequate when phase4 lands cleanly.
# Same architecture, same bundle (distillation_bundle_v4 with the
# English-first tokenizer fix from v9).  Designed to run on GPU
# (validated 19x matmul + 11x mamba speedup vs CPU in benchmark).
PROFILES["hybrid_medium_v10"] = deepcopy(PROFILES["hybrid_medium_v9"])
PROFILES["hybrid_medium_v10"]["warmup_steps"] = 160
PROFILES["hybrid_medium_v10"]["phase_lr_scale"] = {
    "phase1_algorithms": 0.90,
    "phase2_structured": 0.85,
    "phase3_curated_text": 0.40,  # was 0.55 in v9, calmer phase3 learning
    "phase4_instructions": 0.85,  # was 1.00 in v9
    "phase5_verifier": 1.0,
    "phase6_memory": 1.0,
}
PROFILES["hybrid_medium_v10"]["validation_scope"] = (
    "hybrid_medium_v10: Same arch and bundle as v9 (distillation_bundle_v4). "
    "Adjusts learning-rate schedule based on the v9 phase3 plateau: longer "
    "warmup (160), lower phase3 lr_scale (0.40), lower phase4 lr_scale (0.85). "
    "Targets EMA < 6.5 by end of phase3."
)

# ── hybrid_medium_v10_gpu ─────────────────────────────────────────────────
# GPU-tuned variant of v10.  Microbench (GTX 1050 Ti) showed that batch=3
# at d_model=512 leaves bitlinear and matmul in the "GPU loses" regime
# (sub-1× speedup vs CPU) because launch overhead dominates the work.
# At batch=16, the same kernels move into the "GPU dominates" regime
# (matmul 12-19×, bitlinear 1.4×+).  We keep the same step counts —
# each step now processes ~5× the tokens, so total token budget grows
# proportionally and the model sees more data per phase.
#
# Learning-rate tuning: linear scaling rule scaled down (sqrt rule is
# too aggressive at high LR / small models).  4e-4 × sqrt(16/3) ≈ 9.2e-4
# would oscillate; we use 6.5e-4 which empirically lands between the
# two and keeps the v10 LR schedule intent.
#
# Designed to produce wall-clock 3-5× faster than v10 CPU on GTX 1050 Ti
# while training on 5× more tokens per phase — net: comparable token
# throughput, much higher GPU utilization.
PROFILES["hybrid_medium_v10_gpu"] = deepcopy(PROFILES["hybrid_medium_v10"])
# Power envelope tuning history on this GTX 1050 Ti host:
#   batch=16 → host shutdown at phase1→phase2 transition (PSU trip)
#   batch=10 → host shutdown at phase2→phase3 transition (cumulative)
#   batch=8  → current attempt; halves per-step BitLinear count vs 16
# (12 layers × 8 experts × 2 BitLinear = 192 GPU matmuls/step at batch=8
# vs 480+ at batch=16, and per-matmul size is also smaller so peak
# power is roughly 0.4× the original).
PROFILES["hybrid_medium_v10_gpu"]["batch_size"] = 10
PROFILES["hybrid_medium_v10_gpu"]["lr"] = 5.2e-4
PROFILES["hybrid_medium_v10_gpu"]["warmup_steps"] = 200
PROFILES["hybrid_medium_v10_gpu"]["validation_scope"] = (
    "hybrid_medium_v10_gpu: GPU-tuned variant of v10 with batch_size=8 "
    "(batch=10 and 16 both caused PSU shutdown on the GTX 1050 Ti "
    "host).  LR 4.6e-4 with warmup_steps=200.  Same phase_steps as v10."
)

# ── hybrid_v11_colab_t4 ───────────────────────────────────────────────────
# Federated training profile targeting Google Colab Tesla T4 (sm_75, 16GB).
#
# Why a new profile vs scaling v10_gpu:
#   - T4 has 4× the VRAM of GTX 1050 Ti (16GB vs 4GB) → batch can grow
#     until per-step memory peaks at ~12GB (leaving headroom for cuBLAS
#     workspaces + activations + Adam state).
#   - T4 has 4× the FP32 throughput (8.1 TFLOPS vs 2.1) and 16× FP16
#     (65 TFLOPS vs 4) — so per-step wall time at the same batch is ~3-4×
#     faster.  Combined with the bigger batch, real token throughput is
#     ~12-15× the local GPU.
#   - T4 is data-center silicon: real demand paging support, no PSU
#     budget concerns (Google's hardware, not our 500W cheap PSU).
#     The chunked train_loop fix from v10_gpu is still kept but the
#     chunk size can grow back to 8 since trip risk is gone.
#   - sm_75 fully supports __dp4a (INT8 4-element dot product) so the
#     deeper BitLinear path planned in phase 5b unlocks here.
#
# Sizing rationale:
#   batch_size=32 — 3.2× v10_gpu's 10.  At seq_len=512 (default), this is
#                   16K tokens/step.  Peak VRAM ≈ 10-11GB (measured during
#                   smoke).  Leaves 5GB for cuBLAS reuse + safety.
#   lr=8e-4       — sqrt(32/10)×5.2e-4 ≈ 9.3e-4 from linear scaling;
#                   conservative 8e-4 to absorb T4's lower precision in
#                   sparse MoE routing (we don't want oscillation).
#   warmup=400    — 2× v10_gpu warmup, scaled with batch.
#   phase_steps   — keep v10 step counts.  Each step now sees 3.2× more
#                   tokens → effective token budget grows from ~6M to
#                   ~19M per phase.  Still subscale for Chinchilla but
#                   ~10× v10_gpu's signal.
#
# Sessions: Colab Free disconnects at ~12h idle / ~24h running.
#   ─ phase1+2+3 fits in one 8h session (~5h estimated).
#   ─ phase4+5+6+polish fits in a second 6h session.
#   ─ Auto-checkpoint to Drive every 100 steps via colab_bootstrap.
PROFILES["hybrid_v11_colab_t4"] = deepcopy(PROFILES["hybrid_medium_v10_gpu"])
PROFILES["hybrid_v11_colab_t4"]["batch_size"] = 32
PROFILES["hybrid_v11_colab_t4"]["lr"] = 8.0e-4
PROFILES["hybrid_v11_colab_t4"]["warmup_steps"] = 400
# AUDIT #2 (2026-05-16): disable gradient checkpointing.  The 40M model
# at d_model=512, batch=32, seq_len=512 produces ~2.4 GB of activations
# which fits comfortably in T4's 16 GB VRAM (peak ~6 GB total including
# Adam state + cuBLAS workspace).  Gradient checkpointing trades 30%
# extra compute for memory we do not need — a net 1.3x slowdown.  The
# v10 default for d_model >= 256 was conservative for marginal-VRAM
# hosts; on T4 we explicitly opt out.
PROFILES["hybrid_v11_colab_t4"]["use_gradient_checkpointing"] = False
# LEARN A4 (2026-05-16): turn on repetition unlikelihood for phase 3
# (curated text / Cosmopedia / TinyStories).  v10 trained with rul=0.0
# on phase 3 and the resulting model fell into repetition loops
# ("duo duo duo") during inference.  Setting per-phase RUL = 0.05 here
# means the unlikelihood term applies a small penalty to tokens that
# already appear in the recent context — gentle enough not to hurt
# fluency (per Welleck et al. 2020), strong enough to break loops.
PROFILES["hybrid_v11_colab_t4"]["repetition_unlikelihood_scale"] = 0.05
PROFILES["hybrid_v11_colab_t4"]["phase_repetition_unlikelihood_scale"] = {
    "phase1_algorithms":   0.00,   # exact math answers — no penalty
    "phase2_structured":   0.02,   # short extractive QA — small penalty
    "phase3_curated_text": 0.05,   # documents — primary RUL target
    "phase4_instructions": 0.05,   # instructions — moderate penalty
    "phase5_verifier":     0.02,
    "phase6_memory":       0.03,
    "instruction_polish":  0.05,
}
# LEARN A2 (2026-05-16): switch to the v11 curriculum order
# (curated_text → algorithms → structured → instructions → verifier →
# memory).  Strategic Data Ordering (Arxiv 2405.07490) and Efficient
# Pretraining via Curriculum (Arxiv 2506.11300) both show that easier
# distributions first builds better foundational representations for
# under-trained small models.
PROFILES["hybrid_v11_colab_t4"]["curriculum_phase_order"] = "v11"
PROFILES["hybrid_v11_colab_t4"]["validation_scope"] = (
    "hybrid_v11_colab_t4: Federated training on Colab Tesla T4 (sm_75, 16GB). "
    "batch_size=32 (~16K tokens/step), lr=8e-4, warmup=400. "
    "use_gradient_checkpointing=False (1.3x faster — 40M fits T4 VRAM trivially). "
    "Phase-aware repetition unlikelihood (Welleck et al. 2020) breaks "
    "v10's 'duo duo duo' inference loops without hurting fluency. "
    "Same arch/bundle as v10_gpu but ~10x effective token budget per phase "
    "thanks to 4x VRAM headroom.  Designed for 2 Colab sessions: "
    "session 1 = phases 1-3, session 2 = phases 4-6 + polish.  "
    "Auto-checkpoint to Drive every 100 steps."
)

# ── hybrid_v11_colab_a100 ─────────────────────────────────────────────────
# Colab Pro+ variant targeting A100 (sm_80, 40GB).  Same model, much
# bigger batch.  Roughly 5× faster wall-clock than T4 profile.
#
# Sizing:
#   batch_size=64 — 2× the T4 profile; A100 has 2.5× the VRAM and ~5×
#                   the FP32 throughput, so we're VRAM-bound at this
#                   batch with d_model=512 and 12 layers (peak ~22GB).
#   lr=1.1e-3     — sqrt(64/10)×5.2e-4 ≈ 1.3e-3; we use 1.1e-3 since
#                   A100 sustains higher LR without divergence on small
#                   models (tested in the literature for similar configs).
#   warmup=600    — proportional bump.
#   phase_steps   — same as v10, but with batch=64 each step sees 6.4×
#                   more tokens than v10_gpu → ~38M tokens/phase.
#                   Still subscale for 40M-param Chinchilla but unlocks
#                   meaningful generalization signal.
#
# A100 sessions: Colab Pro+ 24h running.  Whole curriculum (1164 steps)
# fits comfortably in one 4-5h session at ~10K tokens/step throughput.
PROFILES["hybrid_v11_colab_a100"] = deepcopy(PROFILES["hybrid_v11_colab_t4"])
PROFILES["hybrid_v11_colab_a100"]["batch_size"] = 64
PROFILES["hybrid_v11_colab_a100"]["lr"] = 1.1e-3
PROFILES["hybrid_v11_colab_a100"]["warmup_steps"] = 600
PROFILES["hybrid_v11_colab_a100"]["validation_scope"] = (
    "hybrid_v11_colab_a100: Federated training on Colab Pro+ A100 (sm_80, 40GB). "
    "batch_size=64 (~32K tokens/step), lr=1.1e-3, warmup=600. "
    "Full curriculum (1164 steps) fits in one 4-5h session.  "
    "Auto-checkpoint to Drive every 50 steps."
)

# ── hybrid_v11_80m base architecture ──────────────────────────────────────
# 80M-param variant of the hybrid architecture for v11.  Same topology
# (Mamba+Attention+MoE hybrid, MoE every 3, attention every 2), scaled
# wider AND deeper:
#   layers      : 12  -> 16   (+33%)
#   d_model     : 512 -> 640  (+25%)
# Total params land at ~80M (per layer cost grows as 1.56× × layer
# count 1.33× = 2.08× → 40M × 2.08 = 83M).
#
# Why scale BOTH instead of just wider or just deeper:
#   * Deeper (+layers) helps sequential reasoning (multi-step inference,
#     consistency over long generations).
#   * Wider (+d_model) helps knowledge density per token (vocabulary
#     coverage, fact storage).
#   * The 2.08× param multiplier matches Chinchilla-optimal for a 1.6B
#     token budget, which is what these profiles target.
#
# MoE is kept at 8 experts × top-2.  Hidden_dim auto-scales to 4×d_model
# (=2560), so each expert grows proportionally — total MoE params grow
# linearly with d_model² × num_experts.
#
# Stability notes for the wider d_model:
#   * MoE routing softmax is more sensitive at d_model > 512; we keep
#     temperature default but add slightly higher routing aux_loss
#     implicitly via the existing MoE balancing loss.
#   * LR is scaled by 1/sqrt(width_ratio) per the standard small-LM
#     transfer rule: 8e-4 × sqrt(512/640) ≈ 7.15e-4 → use 7e-4.
_v11_80m_model_config = {
    "n_heads": 10,           # 640/64 = 10 head_dim=64
    "n_kv_heads": 5,         # halved for GQA
    "sliding_window": 4096,
    "attention_period": 2,
    "attention_slot": 2,
    "use_moe": True,
    "num_experts": 8,
    "num_experts_per_token": 2,
    "moe_period": 3,
    "moe_slot": 3,
    # Cherry-pick #4 (Nemotron K·m invariant): expert FFN intermediate dim.
    # 0 = use historical default of d_model * 4 = 640 * 4 = 2560.
    # Override in variants below (_km_A, _km_B, _km_C) to test the
    # K · m invariant.  See docs/NEMOTRON_KM_INTEGRATION.md.
    "moe_expert_hidden_dim": 0,
    "use_ttt": False,
    "ttt_period": 64,
    "ttt_slot": 63,
    "use_exact_attention_training": True,
    "use_flash_attn": False,
}
# Scale phase_steps proportionally to target ~1.6B tokens (4× v10_gpu
# baseline of ~400M effective).  Phase 3 (Wikipedia/Cosmopedia mass) gets
# the largest bump because that's where coherence is built.
_v11_80m_phase_steps = {
    "phase1_algorithms":   60,    # 2.5× v10 (24)
    "phase2_structured":   40,    # 2.5× v10 (16)
    "phase3_curated_text": 1920,  # 4× v10 (480) — the headline change
    "phase4_instructions": 1120,  # 2.5× v10 (448)
    "phase5_verifier":     80,
    "phase6_memory":       120,
}

# ── hybrid_v11_colab_t4_80m ───────────────────────────────────────────────
# 80M variant tuned for T4 (16GB, sm_75).  Batch is reduced from 32 to
# 24 to fit the larger model + activation memory:
#   weights:  80M × 4 bytes      = 320 MB
#   grads:                       = 320 MB
#   adam (m,v): 80M × 4 × 2      = 640 MB
#   activations: 24 × 512 × 640 × 16 × 6 × 4 = ~3.0 GB
#   cuBLAS workspace + cudart:   = ~1.5 GB
#   total peak:                  = ~5.8 GB (comfortable on T4 16GB)
PROFILES["hybrid_v11_colab_t4_80m"] = deepcopy(PROFILES["hybrid_v11_colab_t4"])
PROFILES["hybrid_v11_colab_t4_80m"]["layers"] = 16
PROFILES["hybrid_v11_colab_t4_80m"]["d_model"] = 640
PROFILES["hybrid_v11_colab_t4_80m"]["model_config"] = deepcopy(_v11_80m_model_config)
PROFILES["hybrid_v11_colab_t4_80m"]["batch_size"] = 24
PROFILES["hybrid_v11_colab_t4_80m"]["lr"] = 7.0e-4
PROFILES["hybrid_v11_colab_t4_80m"]["warmup_steps"] = 500
PROFILES["hybrid_v11_colab_t4_80m"]["phase_steps"] = deepcopy(_v11_80m_phase_steps)
# AUDIT #2 + LEARN A4 propagated from hybrid_v11_colab_t4 (40m).  80M is
# still well under T4 16GB budget (peak ~10 GB with batch=24).
PROFILES["hybrid_v11_colab_t4_80m"]["use_gradient_checkpointing"] = False
PROFILES["hybrid_v11_colab_t4_80m"]["repetition_unlikelihood_scale"] = 0.05
PROFILES["hybrid_v11_colab_t4_80m"]["phase_repetition_unlikelihood_scale"] = {
    "phase1_algorithms":   0.00,
    "phase2_structured":   0.02,
    "phase3_curated_text": 0.05,
    "phase4_instructions": 0.05,
    "phase5_verifier":     0.02,
    "phase6_memory":       0.03,
    "instruction_polish":  0.05,
}
PROFILES["hybrid_v11_colab_t4_80m"]["validation_scope"] = (
    "hybrid_v11_colab_t4_80m: 80M-param hybrid (16 layers, d_model=640) on Colab T4. "
    "batch_size=24 (~12K tokens/step), lr=7e-4, warmup=500. "
    "Targets ~1.6B tokens across phases. Wall time: ~24h split across "
    "2-3 sessions of 8-12h.  Auto-checkpoint to Drive every 100 steps."
)

# ── Nemotron K·m invariant ablation variants (Cherry-pick #4) ─────────────
# Three configurations holding K · m = 5120 fixed (matches baseline above
# at K=2, m=2560).  See OXN/nsos/docs/NEMOTRON_KM_INTEGRATION.md.
# Run all three side-by-side in a smoke ablation to empirically pick the
# winner before committing to the long pre-training run.
# Theory predicts ordering (best to worst): _km_C > _km_A ≈ _km_B > baseline.

# Variant A — more sparse routing.  K doubled, m halved (Principle 5: more
# expert combinations → higher quality at same compute).
PROFILES["hybrid_v11_colab_t4_80m_km_A"] = deepcopy(PROFILES["hybrid_v11_colab_t4_80m"])
PROFILES["hybrid_v11_colab_t4_80m_km_A"]["model_config"] = deepcopy(_v11_80m_model_config)
PROFILES["hybrid_v11_colab_t4_80m_km_A"]["model_config"]["num_experts_per_token"] = 4
PROFILES["hybrid_v11_colab_t4_80m_km_A"]["model_config"]["moe_expert_hidden_dim"] = 1280
# K · m = 4 × 1280 = 5120 (invariant held)
PROFILES["hybrid_v11_colab_t4_80m_km_A"]["validation_scope"] = (
    "Nemotron K·m ablation A: K=4, m=1280, N=8.  K·m=5120 (matches baseline). "
    "Tests Principle 5 (more expert combinations at same compute)."
)

# Variant B — more total experts at same K.  N doubled, m kept, K kept.
# This BREAKS the K·m invariant nominally (still K·m=5120) but adds
# parameter count via more experts.  Mostly tests Principle 5 from the
# other axis (more N, same K).
PROFILES["hybrid_v11_colab_t4_80m_km_B"] = deepcopy(PROFILES["hybrid_v11_colab_t4_80m"])
PROFILES["hybrid_v11_colab_t4_80m_km_B"]["model_config"] = deepcopy(_v11_80m_model_config)
PROFILES["hybrid_v11_colab_t4_80m_km_B"]["model_config"]["num_experts"] = 16
# Default m (0 = dm*4 = 2560), default K=2.  N doubled.
PROFILES["hybrid_v11_colab_t4_80m_km_B"]["validation_scope"] = (
    "Nemotron K·m ablation B: K=2, m=2560, N=16.  More total experts, same K. "
    "Tests Principle 5 (combinatorial expansion via N alone)."
)

# Variant C — combination of A and B.  K doubled, m halved, N doubled.
# Should be the theoretically strongest variant: maximizes combinations
# while keeping per-expert compute small.
PROFILES["hybrid_v11_colab_t4_80m_km_C"] = deepcopy(PROFILES["hybrid_v11_colab_t4_80m"])
PROFILES["hybrid_v11_colab_t4_80m_km_C"]["model_config"] = deepcopy(_v11_80m_model_config)
PROFILES["hybrid_v11_colab_t4_80m_km_C"]["model_config"]["num_experts"] = 16
PROFILES["hybrid_v11_colab_t4_80m_km_C"]["model_config"]["num_experts_per_token"] = 4
PROFILES["hybrid_v11_colab_t4_80m_km_C"]["model_config"]["moe_expert_hidden_dim"] = 1280
# K · m = 4 × 1280 = 5120 (invariant held); N=16 (doubled).
PROFILES["hybrid_v11_colab_t4_80m_km_C"]["validation_scope"] = (
    "Nemotron K·m ablation C: K=4, m=1280, N=16.  Full sparse expansion. "
    "Theory predicts strongest of the three variants."
)

# ── hybrid_rtx2080ti_200m_chinchilla25b ──────────────────────────────────────
# 200M-param hybrid sized for NVIDIA RTX 2080 Ti (11GB VRAM, Turing sm_75).
# Targets ~2.5B training tokens (62% of Chinchilla-optimal for 200M params)
# delivering the meaningful quality jump over 80M needed for first contábil
# design-partner demos.
#
# VRAM math at peak:
#   weights FP32:   200M × 4 bytes              = 800 MB
#   gradients:                                  = 800 MB
#   Adam (m, v):    200M × 4 × 2                = 1.6 GB
#   activations:    batch=12 × seq=512 × d=1024 × 20 × ~6 × 4 bytes
#                   with gradient_checkpointing  ≈ 4.5 GB
#   cuBLAS + cudart workspace                   ≈ 1.5 GB
#   ──────────────────────────────────────────────────────
#   total peak:                                 ≈ 9.2 GB  (fits in 11GB w/ margin)
#
# Wall time @ ~5K tokens/sec on 2080 Ti FP16 with grad checkpoint:
#   2.5B tokens / 5K tok/s = 500K sec ≈ 140h ≈ 5.8 days continuous
#   Realistic at 18h/day (friend uses PC 6h/day): ~8-9 calendar days
#
# Token budget across phases (sum ≈ 2.5B):
#   phase1_algorithms:   320 steps × 12 batch × 160 seq = ~6.1M tokens × loop factor
#   phase2_structured:   240 steps                       = ~4.6M tokens × loop factor
#   phase3_curated_text: 900 steps (heaviest — coherence)= ~17M × loop factor (~1.6B effective)
#   phase4_instructions: 300 steps                       = ~5.7M tokens × loop factor
#   phase5_verifier:     200 steps                       = ~3.8M tokens × loop factor
#   phase6_memory:       320 steps                       = ~6.1M tokens × loop factor
#
# Use this profile when training on the friend's RTX 2080 Ti for the first
# Oxta Contábil production model.  Pair with `--checkpoint-every-steps 100`
# so a crash/power-outage loses at most ~25 min of progress.
_v11_200m_model_config = {
    "n_heads": 16,                  # 1024 / 64 = head_dim 64
    "n_kv_heads": 4,                # GQA 4:1
    "sliding_window": 4096,
    "attention_period": 2,          # 1 attention layer for every 2 layers
    "attention_slot": 2,
    "use_moe": True,
    "num_experts": 8,
    "num_experts_per_token": 2,
    "moe_period": 3,                # 1 MoE layer for every 3 layers
    "moe_slot": 3,
    "moe_expert_hidden_dim": 0,     # default dm * 4 = 4096
    "use_ttt": False,
    "ttt_period": 64,
    "ttt_slot": 63,
    "use_exact_attention_training": True,
    "use_flash_attn": False,        # Turing sm_75 doesn't have Flash Attention
}

_v11_200m_phase_steps = {
    "phase1_algorithms":    320,
    "phase2_structured":    240,
    "phase3_curated_text":  900,    # bulk of curriculum — natural language coherence
    "phase4_instructions":  300,
    "phase5_verifier":      200,
    "phase6_memory":        320,
}

PROFILES["hybrid_rtx2080ti_200m_chinchilla25b"] = deepcopy(PROFILES["hybrid_v11_colab_t4_80m"])
PROFILES["hybrid_rtx2080ti_200m_chinchilla25b"]["layers"] = 20
PROFILES["hybrid_rtx2080ti_200m_chinchilla25b"]["d_model"] = 1024
PROFILES["hybrid_rtx2080ti_200m_chinchilla25b"]["model_config"] = deepcopy(_v11_200m_model_config)
PROFILES["hybrid_rtx2080ti_200m_chinchilla25b"]["batch_size"] = 12  # half of T4 80M (VRAM constraint)
PROFILES["hybrid_rtx2080ti_200m_chinchilla25b"]["lr"] = 5.5e-4      # 7e-4 × sqrt(640/1024) ≈ 5.5e-4
PROFILES["hybrid_rtx2080ti_200m_chinchilla25b"]["warmup_steps"] = 700
PROFILES["hybrid_rtx2080ti_200m_chinchilla25b"]["phase_steps"] = deepcopy(_v11_200m_phase_steps)
# Gradient checkpointing is non-negotiable here — without it the 200M model
# at batch=12 + d_model=1024 spills past 11GB on Turing sm_75.
PROFILES["hybrid_rtx2080ti_200m_chinchilla25b"]["use_gradient_checkpointing"] = True
PROFILES["hybrid_rtx2080ti_200m_chinchilla25b"]["validation_scope"] = (
    "hybrid_rtx2080ti_200m_chinchilla25b: 200M-param hybrid (20 layers, "
    "d_model=1024, MoE-8 top-2) for NVIDIA RTX 2080 Ti 11GB.  batch=12 + "
    "gradient_checkpointing=True.  lr=5.5e-4, warmup=700.  Targets ~2.5B "
    "tokens (62% of Chinchilla-optimal for 200M).  Wall time: ~140h "
    "(~6 days 24/7, ~9 days at 18h/day).  Auto-checkpoint every 100 steps "
    "to /content/drive/MyDrive/oxta_packs/ or local checkpoint dir."
)

# ── hybrid_v11_colab_a100_80m ─────────────────────────────────────────────
# 80M variant on A100 (40GB, sm_80).  Batch doubled to 48 for ~2× wall-time
# speedup vs T4.  Whole curriculum fits in one ~5h A100 session.
PROFILES["hybrid_v11_colab_a100_80m"] = deepcopy(PROFILES["hybrid_v11_colab_t4_80m"])
PROFILES["hybrid_v11_colab_a100_80m"]["batch_size"] = 48
PROFILES["hybrid_v11_colab_a100_80m"]["lr"] = 9.5e-4
PROFILES["hybrid_v11_colab_a100_80m"]["warmup_steps"] = 700
PROFILES["hybrid_v11_colab_a100_80m"]["validation_scope"] = (
    "hybrid_v11_colab_a100_80m: 80M hybrid on Colab Pro+ A100. "
    "batch_size=48, lr=9.5e-4. Full curriculum in one 5-6h session.  "
    "Auto-checkpoint to Drive every 50 steps."
)

# ── hybrid_v11_colab_l4_80m ───────────────────────────────────────────────
# 80M variant on L4 (24GB, sm_89 Ada).  Batch 36 — proportional middle
# ground between T4 and A100. Best value-per-compute-unit at this scale.
PROFILES["hybrid_v11_colab_l4_80m"] = deepcopy(PROFILES["hybrid_v11_colab_t4_80m"])
PROFILES["hybrid_v11_colab_l4_80m"]["batch_size"] = 36
PROFILES["hybrid_v11_colab_l4_80m"]["lr"] = 8.5e-4
PROFILES["hybrid_v11_colab_l4_80m"]["warmup_steps"] = 600
PROFILES["hybrid_v11_colab_l4_80m"]["validation_scope"] = (
    "hybrid_v11_colab_l4_80m: 80M hybrid on Colab L4 (sm_89, 24GB). "
    "batch_size=36, lr=8.5e-4. Best value-per-compute-unit (1.76 units/h). "
    "Full curriculum in ~12h."
)

# ── hybrid_v11_colab_l4 ───────────────────────────────────────────────────
# Colab L4 (sm_89 Ada Lovelace, 24GB VRAM).  L4 is the "middle" GPU on
# Colab — between T4 and A100 — and is the most cost-efficient if the
# user buys compute units (1.76 units/h vs A100's 15 units/h).  Memory
# is ~1.5× T4 so batch can grow modestly; FP32 throughput is ~3× T4 so
# wall time drops accordingly.
#
# Sizing rationale:
#   batch_size=48 — 1.5× T4 profile; L4 has 24GB vs T4's 16GB, so we
#                   stay comfortably below the cuBLAS workspace ceiling.
#   lr=9.5e-4     — sqrt(48/10)×5.2e-4 ≈ 1.14e-3 linear scaling target;
#                   9.5e-4 conservatively absorbs sm_89's higher TFLOPS
#                   without oscillating sparse MoE routing.
#   warmup=500    — between T4 (400) and A100 (600).
PROFILES["hybrid_v11_colab_l4"] = deepcopy(PROFILES["hybrid_v11_colab_t4"])
PROFILES["hybrid_v11_colab_l4"]["batch_size"] = 48
PROFILES["hybrid_v11_colab_l4"]["lr"] = 9.5e-4
PROFILES["hybrid_v11_colab_l4"]["warmup_steps"] = 500
PROFILES["hybrid_v11_colab_l4"]["validation_scope"] = (
    "hybrid_v11_colab_l4: Federated training on Colab L4 (sm_89 Ada, 24GB). "
    "batch_size=48 (~24K tokens/step), lr=9.5e-4, warmup=500. "
    "Best value-per-compute-unit on Colab paid (1.76 units/h)."
)

PROFILES["hybrid_fullstack_pilot"] = deepcopy(PROFILES["small"])
PROFILES["hybrid_fullstack_pilot"]["profile_family"] = "hybrid"
PROFILES["hybrid_fullstack_pilot"]["requested_role"] = "fullstack_integration"
PROFILES["hybrid_fullstack_pilot"]["model_config"] = {
    "n_heads": 8,
    "n_kv_heads": 4,
    "sliding_window": 4096,
    "attention_period": 3,
    "attention_slot": 1,
    "use_moe": True,
    "num_experts": 6,
    "num_experts_per_token": 2,
    "moe_period": 3,
    "moe_slot": 2,
    "use_ttt": True,
    "ttt_period": 3,
    "ttt_slot": 0,
    "use_exact_attention_training": True,
    "use_flash_attn": False,
}
PROFILES["hybrid_fullstack_pilot"]["phase_steps"] = {
    "phase1_algorithms": 40,
    "phase2_structured": 32,
    "phase3_curated_text": 96,
    "phase4_instructions": 48,
    "phase5_verifier": 40,
    "phase6_memory": 64,
}
PROFILES["hybrid_fullstack_pilot"]["phase_lr_scale"] = {
    "phase1_algorithms": 0.90,
    "phase2_structured": 0.90,
    "phase3_curated_text": 0.45,
    "phase4_instructions": 0.75,
    "phase5_verifier": 0.80,
    "phase6_memory": 0.85,
}
PROFILES["hybrid_fullstack_pilot"]["auxiliary_stack"] = {
    "enabled": True,
    "default": {
        "enabled": False,
        "session_adapt": False,
        "reasoning": False,
        "memory": False,
        "reasoning_iterations": 1,
        "reasoning_simulations": 24,
        "memory_blend": 0.35,
        "every_steps": 1,
        "prompt_max_tokens": 96,
        "answer_max_tokens": 24,
    },
    "phase_overrides": {
        "phase1_algorithms": {
            "enabled": True,
            "session_adapt": True,
            "reasoning": True,
            "memory": False,
            "reasoning_iterations": 1,
            "reasoning_simulations": 8,
            "every_steps": 2,
            "prompt_max_tokens": 64,
            "answer_max_tokens": 12,
        },
        "phase2_structured": {
            "enabled": True,
            "session_adapt": True,
            "reasoning": True,
            "memory": True,
            "reasoning_iterations": 1,
            "reasoning_simulations": 10,
            "memory_blend": 0.25,
            "every_steps": 2,
            "prompt_max_tokens": 96,
            "answer_max_tokens": 12,
        },
        "phase4_instructions": {
            "enabled": True,
            "session_adapt": True,
            "reasoning": True,
            "memory": False,
            "reasoning_iterations": 1,
            "reasoning_simulations": 10,
            "every_steps": 3,
            "prompt_max_tokens": 96,
            "answer_max_tokens": 20,
        },
        "phase5_verifier": {
            "enabled": True,
            "session_adapt": True,
            "reasoning": True,
            "memory": False,
            "reasoning_iterations": 1,
            "reasoning_simulations": 12,
            "every_steps": 2,
            "prompt_max_tokens": 80,
            "answer_max_tokens": 16,
        },
        "phase6_memory": {
            "enabled": True,
            "session_adapt": True,
            "reasoning": True,
            "memory": True,
            "reasoning_iterations": 1,
            "reasoning_simulations": 12,
            "memory_blend": 0.35,
            "every_steps": 3,
            "prompt_max_tokens": 128,
            "answer_max_tokens": 24,
        },
    },
}
PROFILES["hybrid_fullstack_pilot"]["validation_scope"] = (
    "Validates the full hybrid pilot stack with attention, sparse MoE, TTT, "
    "reasoning-assisted adaptation, and external memory hooks in the main training path."
)

PROFILES["hybrid_core_v25"] = deepcopy(PROFILES["small"])
PROFILES["hybrid_core_v25"]["profile_family"] = "hybrid"
PROFILES["hybrid_core_v25"]["requested_role"] = "hybrid_core_ramp"
PROFILES["hybrid_core_v25"]["model_config"] = {
    "n_heads": 8,
    "n_kv_heads": 4,
    "sliding_window": 4096,
    "attention_period": 3,
    "attention_slot": 1,
    "use_moe": True,
    "num_experts": 6,
    "num_experts_per_token": 2,
    "moe_period": 3,
    "moe_slot": 2,
    "use_ttt": False,
    "ttt_period": 64,
    "ttt_slot": 63,
    "use_exact_attention_training": True,
    "use_flash_attn": False,
}
PROFILES["hybrid_core_v25"]["phase_lr_scale"] = {
    "phase1_algorithms": 0.95,
    "phase2_structured": 0.95,
    "phase3_curated_text": 0.50,
    "phase4_instructions": 0.85,
    "phase5_verifier": 0.90,
    "phase6_memory": 0.90,
}
PROFILES["hybrid_core_v25"]["auxiliary_stack"] = {
    "enabled": False,
}
PROFILES["hybrid_core_v25"]["validation_scope"] = (
    "Validates the first hybrid product ramp with attention + sparse MoE + Mamba, "
    "without TTT or auxiliary reasoning/memory."
)

PROFILES["hybrid_memory_reason_v26"] = deepcopy(PROFILES["hybrid_core_v25"])
PROFILES["hybrid_memory_reason_v26"]["requested_role"] = "hybrid_memory_reason_ramp"
PROFILES["hybrid_memory_reason_v26"]["auxiliary_stack"] = {
    "enabled": True,
    "default": {
        "enabled": False,
        "session_adapt": False,
        "reasoning": False,
        "memory": False,
        "reasoning_iterations": 1,
        "reasoning_simulations": 6,
        "memory_blend": 0.20,
        "every_steps": 4,
        "prompt_max_tokens": 64,
        "answer_max_tokens": 12,
    },
    "phase_overrides": {
        "phase5_verifier": {
            "enabled": True,
            "session_adapt": False,
            "reasoning": True,
            "memory": False,
            "reasoning_iterations": 1,
            "reasoning_simulations": 6,
            "every_steps": 4,
            "prompt_max_tokens": 64,
            "answer_max_tokens": 12,
        },
        "phase6_memory": {
            "enabled": True,
            "session_adapt": False,
            "reasoning": False,
            "memory": True,
            "reasoning_iterations": 1,
            "reasoning_simulations": 0,
            "memory_blend": 0.22,
            "every_steps": 3,
            "prompt_max_tokens": 96,
            "answer_max_tokens": 16,
        },
    },
}
PROFILES["hybrid_memory_reason_v26"]["validation_scope"] = (
    "Validates hybrid attention + MoE + Mamba with light verifier reasoning in phase5 "
    "and light external-memory blending in phase6."
)

PROFILES["hybrid_ttt_tail_v27"] = deepcopy(PROFILES["hybrid_memory_reason_v26"])
PROFILES["hybrid_ttt_tail_v27"]["requested_role"] = "hybrid_ttt_tail_ramp"
PROFILES["hybrid_ttt_tail_v27"]["model_config"] = deepcopy(
    PROFILES["hybrid_memory_reason_v26"]["model_config"]
)
PROFILES["hybrid_ttt_tail_v27"]["model_config"]["use_ttt"] = True
PROFILES["hybrid_ttt_tail_v27"]["model_config"]["ttt_period"] = 3
PROFILES["hybrid_ttt_tail_v27"]["model_config"]["ttt_slot"] = 0
PROFILES["hybrid_ttt_tail_v27"]["auxiliary_stack"] = {
    "enabled": True,
    "default": {
        "enabled": False,
        "session_adapt": False,
        "reasoning": False,
        "memory": False,
        "reasoning_iterations": 1,
        "reasoning_simulations": 8,
        "memory_blend": 0.30,
        "every_steps": 3,
        "prompt_max_tokens": 96,
        "answer_max_tokens": 16,
    },
    "phase_overrides": {
        "phase5_verifier": {
            "enabled": True,
            "session_adapt": False,
            "reasoning": True,
            "memory": False,
            "reasoning_iterations": 1,
            "reasoning_simulations": 10,
            "every_steps": 2,
            "prompt_max_tokens": 80,
            "answer_max_tokens": 16,
        },
        "phase6_memory": {
            "enabled": True,
            "session_adapt": True,
            "reasoning": True,
            "memory": True,
            "reasoning_iterations": 1,
            "reasoning_simulations": 12,
            "memory_blend": 0.35,
            "every_steps": 2,
            "prompt_max_tokens": 128,
            "answer_max_tokens": 24,
        },
        "instruction_polish": {
            "enabled": True,
            "session_adapt": True,
            "reasoning": False,
            "memory": False,
            "reasoning_iterations": 0,
            "reasoning_simulations": 0,
            "every_steps": 2,
            "prompt_max_tokens": 96,
            "answer_max_tokens": 16,
        },
    },
}
PROFILES["hybrid_ttt_tail_v27"]["validation_scope"] = (
    "Validates hybrid attention + MoE + Mamba with reasoning/memory late and TTT introduced only "
    "at the tail of the curriculum."
)

PROFILES["small"]["legacy_alias_for"] = "mamba_small"
PROFILES["pilot"]["legacy_alias_for"] = "hybrid_pilot"

OFFICIAL_SUPPORT_MATRIX = {
    "cpu_float": "validated",
    "cpu_packed": "validated",
    "gpu_float": "validated_hot_path",
    "gpu_packed": "not_yet_native",
    "ttt": "research_only",
}

RELEASE_GATE_THRESHOLDS = {
    "phase1_algorithms": {
        "first_token_accuracy_min": 0.20,
        "exact_accuracy_min": 0.25,
        "answer_loss_max": 4.50,
    },
    "phase2_structured": {
        "first_token_accuracy_min": 0.15,
        "teacher_token_accuracy_min": 0.35,
        "answer_loss_max": 4.75,
    },
    "phase3_curated_text": {
        "heldout_loss_max": 7.10,
    },
    "phase4_instructions": {
        "first_token_accuracy_min": 0.25,
        "probe_prefix_match_ratio_min": 0.45,
        "answer_loss_max": 6.25,
    },
    "phase5_verifier": {
        "first_token_accuracy_min": 0.17,
        "teacher_token_accuracy_min": 0.35,
        "probe_prefix_match_ratio_min": 0.45,
        "answer_loss_max": 3.30,
    },
    "phase6_memory": {
        "first_token_accuracy_min": 0.17,
        "teacher_token_accuracy_min": 0.35,
        "probe_prefix_match_ratio_min": 0.40,
        "answer_loss_max": 4.80,
    },
}

_START_BLOCKLIST_CACHE: Dict[tuple[int, int], set[int]] = {}
_TASK_START_BLOCKLIST_CACHE: Dict[tuple[int, int, str], set[int]] = {}
PHASE_FAMILIES = {
    "phase1_algorithms": "algorithmic",
    "phase2_structured": "structured",
    "phase3_curated_text": "text",
    "phase4_instructions": "instruction",
    "phase5_verifier": "verifier",
    "phase6_memory": "memory",
}
PHASE_REPLAY_CONFIG = {
    "phase1_algorithms": {"families": [], "ratio_scale": 0.0},
    "phase2_structured": {"families": ["algorithmic"], "ratio_scale": 0.35},
    "phase3_curated_text": {"families": [], "ratio_scale": 0.0},
    "phase4_instructions": {"families": [], "ratio_scale": 0.0},
    "phase5_verifier": {"families": ["algorithmic", "structured"], "ratio_scale": 1.0},
    "phase6_memory": {"families": [], "ratio_scale": 0.0},
}
NATURAL_LANGUAGE_TASKS = {
    "summarize",
    "rewrite",
    "translate",
    "translate_pt",
    "translate_en",
    "explain_code",
    "extract_fact",
    "direct_recall",
    "overwrite_recall",
    "pair_recall",
    "fact_table",
    "conversation_recall",
    "story_recall",
    "evidence_extract",
}
CONTAMINATED_START_TEXTS = {
    "status",
    "error",
    "warning",
    "agent",
    "cache",
    "edge",
    "train",
    ",pt,",
    "pt",
    "en",
}
GENERATION_REPETITION_PENALTY = 1.08
GENERATION_NO_REPEAT_NGRAM = 3
SHORT_FORM_REPETITION_PENALTY = 1.02
SHORT_FORM_NO_REPEAT_NGRAM = 2
LOWER_SINGLE_TOKEN_TASKS = {
    "log_extract",
    "config_lookup",
    "direct_recall",
    "overwrite_recall",
    "pair_recall",
    "fact_table",
    "conversation_recall",
    "story_recall",
    "evidence_extract",
}
UPPER_LABEL_TASKS = {
    "parity_label",
    "compare_label",
    "count_label",
    "boolean_gate",
}
INTEGER_START_TASKS = {
    "math_small",
    "dsl",
    "code_output",
    "math_exact",
    "math_word_problem",
}
BINARY_START_TASKS = {
    "binary_add",
    "circuit",
}


def resolve_profile(profile_name: str) -> tuple[str, Dict]:
    canonical = PROFILE_ALIASES.get(profile_name, profile_name)
    profile = deepcopy(PROFILES[canonical])
    profile["requested_profile"] = profile_name
    profile["canonical_profile"] = canonical
    profile["legacy_alias_used"] = canonical != profile_name
    return canonical, profile


def layer_matches_schedule(layer_one_based: int, period: int, slot: int) -> bool:
    period = max(int(period), 1)
    slot = max(0, min(int(slot), period - 1))
    return ((layer_one_based - 1) % period) == slot


def schedule_layer_indices(total_layers: int, enabled: bool, period: int, slot: int) -> List[int]:
    if not enabled:
        return []
    indices: List[int] = []
    for layer_one_based in range(1, total_layers + 1):
        if total_layers >= max(int(period), 1) and layer_matches_schedule(layer_one_based, period, slot):
            indices.append(layer_one_based)
    return indices


def build_model_config(nsos, profile: Dict, vocab_size: int, device) -> object:
    config = nsos.ModelConfig()
    config.num_layers = int(profile["layers"])
    config.d_model = int(profile["d_model"])
    config.vocab_size = int(vocab_size)
    config.use_cuda = device == nsos.Device.GPU
    config.use_gradient_checkpointing = bool(
        profile.get("use_gradient_checkpointing", config.d_model >= 256)
    )
    config.dropout = float(profile.get("dropout", 0.05 if config.d_model >= 256 else 0.02))
    for key, value in profile.get("model_config", {}).items():
        setattr(config, key, value)
    config.default_batch_size = int(profile["batch_size"])
    config.max_context_tokens = max(int(profile["seq_len"]) * 4, 1024)
    return config


def model_config_to_dict(config) -> Dict[str, int | bool | float | str]:
    return {
        "num_layers": int(config.num_layers),
        "d_model": int(config.d_model),
        "vocab_size": int(config.vocab_size),
        "n_heads": int(config.n_heads),
        "n_kv_heads": int(config.n_kv_heads),
        "sliding_window": int(config.sliding_window),
        "attention_period": int(config.attention_period),
        "attention_slot": int(config.attention_slot),
        "num_experts": int(config.num_experts),
        "num_experts_per_token": int(config.num_experts_per_token),
        "use_moe": bool(config.use_moe),
        "moe_period": int(config.moe_period),
        "moe_slot": int(config.moe_slot),
        "use_ttt": bool(config.use_ttt),
        "ttt_period": int(config.ttt_period),
        "ttt_slot": int(config.ttt_slot),
        "use_gradient_checkpointing": bool(config.use_gradient_checkpointing),
        "dropout": float(config.dropout),
        "mcts_simulations": int(config.mcts_simulations),
        "mcts_depth": int(config.mcts_depth),
        "checkpoint_path": str(config.checkpoint_path),
        "max_context_tokens": int(config.max_context_tokens),
        "default_batch_size": int(config.default_batch_size),
        "use_cuda": bool(config.use_cuda),
        "use_exact_attention_training": bool(config.use_exact_attention_training),
        "use_flash_attn": bool(config.use_flash_attn),
    }


def build_effective_schedule(profile: Dict, model_config: Dict[str, int | bool | float | str]) -> Dict[str, object]:
    total_layers = int(model_config["num_layers"])
    attention_layers = schedule_layer_indices(
        total_layers,
        True,
        int(model_config["attention_period"]),
        int(model_config["attention_slot"]),
    )
    moe_layers = schedule_layer_indices(
        total_layers,
        bool(model_config["use_moe"]),
        int(model_config["moe_period"]),
        int(model_config["moe_slot"]),
    )
    ttt_layers = schedule_layer_indices(
        total_layers,
        bool(model_config["use_ttt"]),
        int(model_config["ttt_period"]),
        int(model_config["ttt_slot"]),
    )
    dominant_stack = "mamba"
    if attention_layers or moe_layers or ttt_layers:
        dominant_stack = "hybrid"
    if not attention_layers and not moe_layers and not ttt_layers:
        dominant_stack = "mamba_only"
    return {
        "profile_family": profile.get("profile_family", "unknown"),
        "dominant_stack": dominant_stack,
        "attention_layers": attention_layers,
        "moe_layers": moe_layers,
        "ttt_layers": ttt_layers,
        "validation_scope": profile.get("validation_scope", ""),
        "feature_status": {
            "attention": "product" if attention_layers else "inactive",
            "moe": "product" if moe_layers else "inactive",
            "ttt": "research_only" if ttt_layers or bool(model_config["use_ttt"]) else "inactive",
        },
    }


def compute_release_candidate_score(metrics_by_phase: Dict[str, Dict[str, float]]) -> float:
    total = 0.0
    for phase_name, metrics in metrics_by_phase.items():
        weight = 1.0
        if phase_name == "phase3_curated_text":
            weight = 1.40
        elif phase_name in {"phase1_algorithms", "phase2_structured"}:
            weight = 1.10
        elif phase_name in {"phase4_instructions", "phase5_verifier", "phase6_memory"}:
            weight = 1.25
        total += weight * (
            1.6 * float(metrics.get("exact_accuracy", 0.0))
            + 1.3 * float(metrics.get("probe_exact_match", 0.0))
            + 1.2 * float(metrics.get("probe_prefix_match_ratio", 0.0))
            + 0.9 * float(metrics.get("first_token_accuracy", 0.0))
            + 1.4 * float(metrics.get("teacher_token_accuracy", 0.0))
            - 0.55 * float(metrics.get("answer_loss", 0.0))
            - 0.85 * float(metrics.get("heldout_loss", 0.0))
            - 1.2 * float(metrics.get("probe_repetition_penalty", 0.0))
        )
    return total


def evaluate_release_gate(metrics_by_phase: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    failures: List[Dict[str, object]] = []
    for phase_name, thresholds in RELEASE_GATE_THRESHOLDS.items():
        metrics = metrics_by_phase.get(phase_name, {})
        for metric_name, threshold in thresholds.items():
            value = float(metrics.get(metric_name.replace("_min", "").replace("_max", ""), metrics.get(metric_name, 0.0)))
            if metric_name.endswith("_min"):
                if value < float(threshold):
                    failures.append(
                        {
                            "phase": phase_name,
                            "metric": metric_name[:-4],
                            "expected": f">={threshold}",
                            "actual": value,
                        }
                    )
            elif metric_name.endswith("_max"):
                if value > float(threshold):
                    failures.append(
                        {
                            "phase": phase_name,
                            "metric": metric_name[:-4],
                            "expected": f"<={threshold}",
                            "actual": value,
                        }
                    )
    return {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }


def phase_repetition_scale(profile: Dict, phase_name: str) -> float:
    # Diagnostic/override knob: NSOS_RUL_SCALE forces the repetition-unlikelihood
    # scale for ALL phases (e.g. "0" to disable it).  RUL runs a per-step,
    # per-sample host loop over the full softmax (probs.cpu() + O(seq*neg*vocab))
    # — setting it to 0 isolates how much of the step time is that host path.
    env = os.environ.get("NSOS_RUL_SCALE")
    if env is not None and env != "":
        try:
            return float(env)
        except ValueError:
            pass
    phase_scales = profile.get("phase_repetition_unlikelihood_scale", {})
    return float(phase_scales.get(phase_name, profile.get("repetition_unlikelihood_scale", 0.0)))


def resolve_auxiliary_stack_config(profile: Dict, phase_name: str) -> Dict[str, object]:
    profile_cfg = profile.get("auxiliary_stack", {})
    if not profile_cfg or not bool(profile_cfg.get("enabled", False)):
        return {
            "enabled": False,
            "session_adapt": False,
            "reasoning": False,
            "memory": False,
            "reasoning_iterations": 0,
            "reasoning_simulations": 0,
            "memory_blend": 0.0,
            "every_steps": 1,
            "prompt_max_tokens": 0,
            "answer_max_tokens": 0,
        }

    merged = {
        "enabled": False,
        "session_adapt": False,
        "reasoning": False,
        "memory": False,
        "reasoning_iterations": 1,
        "reasoning_simulations": 24,
        "memory_blend": 0.35,
        "every_steps": 1,
        "prompt_max_tokens": 96,
        "answer_max_tokens": 24,
    }
    merged.update(profile_cfg.get("default", {}))
    merged.update(profile_cfg.get("phase_overrides", {}).get(phase_name, {}))
    enabled = bool(profile_cfg.get("enabled", False) and merged.get("enabled", True))
    return {
        "enabled": enabled,
        "session_adapt": bool(enabled and merged.get("session_adapt", False)),
        "reasoning": bool(enabled and merged.get("reasoning", False)),
        "memory": bool(enabled and merged.get("memory", False)),
        "reasoning_iterations": int(merged.get("reasoning_iterations", 1)) if enabled else 0,
        "reasoning_simulations": int(merged.get("reasoning_simulations", 24)) if enabled else 0,
        "memory_blend": float(merged.get("memory_blend", 0.35)) if enabled else 0.0,
        "every_steps": max(1, int(merged.get("every_steps", 1))),
        "prompt_max_tokens": int(merged.get("prompt_max_tokens", 96)) if enabled else 0,
        "answer_max_tokens": int(merged.get("answer_max_tokens", 24)) if enabled else 0,
    }


def apply_auxiliary_stack_schedule(trainer, aux_cfg: Dict[str, object]) -> None:
    trainer.phase_scheduler.auxiliary_stack_enabled = bool(aux_cfg.get("enabled", False))
    trainer.phase_scheduler.auxiliary_session_adapt_enabled = bool(
        aux_cfg.get("session_adapt", False)
    )
    trainer.phase_scheduler.auxiliary_reasoning_enabled = bool(aux_cfg.get("reasoning", False))
    trainer.phase_scheduler.auxiliary_memory_enabled = bool(aux_cfg.get("memory", False))
    trainer.phase_scheduler.auxiliary_reasoning_iterations = int(
        aux_cfg.get("reasoning_iterations", 0)
    )
    trainer.phase_scheduler.auxiliary_reasoning_simulations = int(
        aux_cfg.get("reasoning_simulations", 0)
    )
    trainer.phase_scheduler.auxiliary_memory_blend = float(aux_cfg.get("memory_blend", 0.0))
    trainer.phase_scheduler.auxiliary_every_steps = int(aux_cfg.get("every_steps", 1))
    trainer.phase_scheduler.auxiliary_prompt_max_tokens = int(
        aux_cfg.get("prompt_max_tokens", 0)
    )
    trainer.phase_scheduler.auxiliary_answer_max_tokens = int(
        aux_cfg.get("answer_max_tokens", 0)
    )


def auxiliary_stats_to_dict(stats) -> Dict[str, float | int]:
    return {
        "bucket_count": int(getattr(stats, "bucket_count", 0)),
        "due_count": int(getattr(stats, "due_count", 0)),
        "applied_count": int(getattr(stats, "applied_count", 0)),
        "reasoning_count": int(getattr(stats, "reasoning_count", 0)),
        "memory_count": int(getattr(stats, "memory_count", 0)),
        "session_adapt_count": int(getattr(stats, "session_adapt_count", 0)),
        "sample_count": int(getattr(stats, "sample_count", 0)),
        "prompt_tokens": int(getattr(stats, "prompt_tokens", 0)),
        "answer_tokens": int(getattr(stats, "answer_tokens", 0)),
        "prompt_state_norm": float(getattr(stats, "prompt_state_norm", 0.0)),
        "target_state_norm": float(getattr(stats, "target_state_norm", 0.0)),
        "reason_delta_norm": float(getattr(stats, "reason_delta_norm", 0.0)),
        "reason_cosine": float(getattr(stats, "reason_cosine", 0.0)),
        "memory_delta_norm": float(getattr(stats, "memory_delta_norm", 0.0)),
        "memory_cosine": float(getattr(stats, "memory_cosine", 0.0)),
        "final_target_delta_norm": float(getattr(stats, "final_target_delta_norm", 0.0)),
    }


def merge_auxiliary_metrics(total: Dict[str, float], current: Dict[str, float | int]) -> None:
    for key, value in current.items():
        total[key] = float(total.get(key, 0.0)) + float(value)


def finalize_auxiliary_metrics(total: Dict[str, float]) -> Dict[str, float]:
    result = dict(total)
    applied = max(int(result.get("applied_count", 0)), 1)
    reasoning = max(int(result.get("reasoning_count", 0)), 1)
    memory = max(int(result.get("memory_count", 0)), 1)
    if result.get("applied_count", 0) > 0:
        result["prompt_state_norm"] /= applied
        result["target_state_norm"] /= applied
        result["final_target_delta_norm"] /= applied
    if result.get("reasoning_count", 0) > 0:
        result["reason_delta_norm"] /= reasoning
        result["reason_cosine"] /= reasoning
    if result.get("memory_count", 0) > 0:
        result["memory_delta_norm"] /= memory
        result["memory_cosine"] /= memory
    return result


def format_auxiliary_metrics(metrics: Dict[str, float]) -> str:
    due = int(metrics.get("due_count", 0))
    applied = int(metrics.get("applied_count", 0))
    reasoning = int(metrics.get("reasoning_count", 0))
    memory = int(metrics.get("memory_count", 0))
    session_adapt = int(metrics.get("session_adapt_count", 0))
    prompt_tokens = int(metrics.get("prompt_tokens", 0))
    answer_tokens = int(metrics.get("answer_tokens", 0))
    return (
        f"due={due} applied={applied} reason={reasoning} mem={memory} ttt={session_adapt} "
        f"prompt_tok={prompt_tokens} answer_tok={answer_tokens} "
        f"reason_d={metrics.get('reason_delta_norm', 0.0):.3f} "
        f"reason_cos={metrics.get('reason_cosine', 0.0):.3f} "
        f"mem_d={metrics.get('memory_delta_norm', 0.0):.3f} "
        f"mem_cos={metrics.get('memory_cosine', 0.0):.3f} "
        f"final_d={metrics.get('final_target_delta_norm', 0.0):.3f}"
    )


def generation_guard_profile(task_kind: str) -> tuple[float, int]:
    if task_kind in LOWER_SINGLE_TOKEN_TASKS:
        return SHORT_FORM_REPETITION_PENALTY, SHORT_FORM_NO_REPEAT_NGRAM
    if task_kind in UPPER_LABEL_TASKS:
        return SHORT_FORM_REPETITION_PENALTY, SHORT_FORM_NO_REPEAT_NGRAM
    if task_kind in INTEGER_START_TASKS:
        return SHORT_FORM_REPETITION_PENALTY, SHORT_FORM_NO_REPEAT_NGRAM
    if task_kind in BINARY_START_TASKS:
        return SHORT_FORM_REPETITION_PENALTY, SHORT_FORM_NO_REPEAT_NGRAM
    return GENERATION_REPETITION_PENALTY, GENERATION_NO_REPEAT_NGRAM


class RunLogger:
    def __init__(self, mode: str, session_log_path: Path | None):
        resolved_mode = mode
        if resolved_mode == "auto":
            resolved_mode = "tqdm" if tqdm is not None and sys.stderr.isatty() else "plain"
        if resolved_mode == "tqdm" and (tqdm is None or not sys.stderr.isatty()):
            resolved_mode = "plain"
        self.mode = resolved_mode
        self._file: TextIO | None = None
        if session_log_path is not None:
            session_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = session_log_path.open("w", encoding="utf-8")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def log(self, message: str) -> None:
        if self._file is not None:
            self._file.write(message + "\n")
            self._file.flush()
        if self.mode == "tqdm" and tqdm is not None:
            tqdm.write(message)
        else:
            print(message, flush=True)

    def make_progress(self, total: int, desc: str):
        if self.mode != "tqdm" or tqdm is None:
            return None
        return tqdm(total=total, desc=desc, dynamic_ncols=True, leave=True)


def estimate_current_lr(trainer) -> float:
    warmup_steps = max(int(trainer.warmup_steps), 1)
    step = int(trainer.global_step_count)
    if step <= 0:
        return 0.0
    if step <= warmup_steps:
        return float(trainer.learning_rate) * step / warmup_steps
    total_steps = max(int(trainer.total_training_steps), warmup_steps + 1)
    decay_span = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_span, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = float(trainer.learning_rate) * float(trainer.min_learning_rate_scale)
    return floor + (float(trainer.learning_rate) - floor) * cosine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NSOS on the staged curriculum.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "curriculum_bundle",
        help="Directory that stores the curriculum bundle and tokenizer artifacts.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "curriculum_runs" / "latest",
        help="Directory for checkpoints and training metrics.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="Explicit build directory containing nsos_ext.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="mamba_small",
        help="Training profile.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Execution device.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Bundle seed.")
    parser.add_argument("--rebuild-curriculum", action="store_true", help="Regenerate curriculum and tokenizer.")
    parser.add_argument(
        "--disable-qat",
        action="store_true",
        help="Disable progressive QAT for this run even if the selected profile enables it.",
    )
    parser.add_argument(
        "--override-phase-steps",
        type=int,
        default=0,
        help="If > 0, replace all per-phase max_steps with this value.",
    )
    parser.add_argument(
        "--phase-eval-samples",
        type=int,
        default=8,
        help="Held-out samples per phase used during in-training evaluation.",
    )
    parser.add_argument(
        "--phase-eval-mode",
        choices=["fast", "full"],
        default="fast",
        help="Fast masked-answer eval or full greedy exact-match eval after each phase.",
    )
    parser.add_argument(
        "--phase-exact-samples",
        type=int,
        default=0,
        help="Number of held-out samples used for greedy exact-match when eval mode is full.",
    )
    parser.add_argument(
        "--fast-exact-samples",
        type=int,
        default=0,
        help="Greedy exact-match samples in fast eval mode. 0 keeps exact-match out of fast gates.",
    )
    parser.add_argument(
        "--generation-probe-samples",
        type=int,
        default=0,
        help="Rows used by the generation probe in evaluate_phase. 0 disables the probe.",
    )
    parser.add_argument(
        "--eval-masked-batch-size",
        type=int,
        default=4,
        help="Batch size for masked supervised evaluation when forward_ids_batch is available.",
    )
    parser.add_argument(
        "--text-loss-max-windows",
        type=int,
        default=8,
        help="Maximum text-loss windows per phase evaluation.",
    )
    parser.add_argument(
        "--global-suite-samples-per-phase",
        type=int,
        default=8,
        help="Held-out rows per phase used to score the global champion checkpoint.",
    )
    parser.add_argument(
        "--global-suite-exact-samples",
        type=int,
        default=4,
        help="Greedy exact-match samples per phase used in global champion evaluation.",
    )
    parser.add_argument(
        "--global-suite-every-phases",
        type=int,
        default=0,
        help="Run global champion evaluation every N phases instead of after every phase.",
    )
    parser.add_argument(
        "--replay-ratio",
        type=float,
        default=0.25,
        help="Fraction of previous supervised rows replayed into later supervised phases.",
    )
    parser.add_argument(
        "--final-consolidation-steps",
        type=int,
        default=0,
        help="Optional final supervised replay pass over all non-document phases.",
    )
    parser.add_argument(
        "--resume-model",
        type=Path,
        default=None,
        help="Optional checkpoint/model to load before training continues.",
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=0,
        help="If > 0, save rolling checkpoints every N supervised/text steps.",
    )
    parser.add_argument(
        "--phase-best-eval-every-steps",
        type=int,
        default=0,
        help="If > 0, run in-phase eval every N steps and keep the best checkpoint.",
    )
    parser.add_argument(
        "--log-ema-beta",
        type=float,
        default=0.9,
        help="EMA beta used for smoothed loss logging.",
    )
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=5,
        help="Plain-mode logging cadence for per-step loss updates.",
    )
    parser.add_argument(
        "--progress-mode",
        choices=["auto", "tqdm", "plain"],
        default="auto",
        help="Inline progress rendering mode.",
    )
    parser.add_argument(
        "--session-log",
        type=Path,
        default=None,
        help="Optional session log written directly by the trainer script.",
    )
    parser.add_argument(
        "--enable-layer-audit",
        action="store_true",
        help="Attach the C++ layer audit collector and write per-phase audit snapshots.",
    )
    parser.add_argument(
        "--layer-audit-out",
        type=Path,
        default=None,
        help="Layer audit JSON output path. Defaults to <run-dir>/layer_audit.json.",
    )
    parser.add_argument(
        "--audit-summary-out",
        type=Path,
        default=None,
        help="Layer audit summary JSON output path. Defaults to <run-dir>/layer_audit_summary.json.",
    )
    parser.add_argument(
        "--audit-min-layer-coverage",
        type=int,
        default=0,
        help="Minimum audited model layers per audited phase. 0 means model_config.num_layers.",
    )
    parser.add_argument(
        "--audit-router-entropy-min",
        type=float,
        default=0.0,
        help="Minimum MoE router entropy when router records exist.",
    )
    parser.add_argument(
        "--audit-router-entropy-max",
        type=float,
        default=0.0,
        help="Maximum MoE router entropy. 0 means auto log2(num_experts)+epsilon.",
    )
    parser.add_argument(
        "--audit-max-reload-drift",
        type=float,
        default=1.0e-5,
        help="Maximum allowed absolute logit drift between pre/post reload probes.",
    )
    parser.add_argument(
        "--audit-summary-only",
        action="store_true",
        help="Keep per-phase audit summaries but omit raw layer records from audit JSON.",
    )
    parser.add_argument(
        "--audit-record-sample-rate",
        type=int,
        default=1,
        help="Collect forward/backward tensor stats every N observed records; router records are always collected.",
    )
    parser.add_argument(
        "--audit-max-records-per-phase",
        type=int,
        default=0,
        help="Maximum raw layer records stored per phase. 0 means unlimited.",
    )
    parser.add_argument(
        "--audit-store-token-contexts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store raw token contexts in audit JSON. Phase summaries always keep token-context counts.",
    )
    parser.add_argument(
        "--holdout-files",
        type=Path,
        nargs="*",
        default=[
            Path(__file__).resolve().parents[1] / "benchmarks" / "nsos_micro_suite.jsonl",
            Path(__file__).resolve().parents[1] / "benchmarks" / "nsos_eval_suite.jsonl",
        ],
        help="Official holdout JSONL files evaluated at the end of the run.",
    )
    parser.add_argument(
        "--holdout-eval-mode",
        choices=["fast", "full"],
        default="fast",
        help="Evaluation mode for official holdout files.",
    )
    parser.add_argument(
        "--holdout-exact-samples",
        type=int,
        default=0,
        help="Greedy exact-match samples per official holdout file.",
    )
    parser.add_argument(
        "--holdout-every-phases",
        type=int,
        default=0,
        help=(
            "If > 0, evaluate official holdouts every N phases and keep "
            "a capacity champion checkpoint. 0 evaluates holdouts only at the end."
        ),
    )
    parser.add_argument(
        "--select-final-by-holdout",
        action="store_true",
        help="Load the best official-holdout capacity champion before final holdout/audit gates.",
    )
    return parser.parse_args()


def detect_build_dir(explicit: Path | None) -> Path:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.extend([explicit, explicit / "Release"])
    repo_root = Path(__file__).resolve().parents[3]
    nsos_root = repo_root / "OXN" / "nsos"
    # Also search the Colab build dir produced by colab/colab_bootstrap.py
    for name in ["build_cuda129", "build_v1", "build_full", "build_codex",
                 "build", "build-colab", "build-cuda-validation"]:
        candidates.extend([nsos_root / name / "Release", nsos_root / name])

    # Accept both Windows (.pyd) and Linux (.so) Python extension files
    # so the same train_curriculum.py works on the local 1050ti host AND
    # on Colab Linux without changing the search logic.
    patterns = ("nsos_ext*.pyd", "nsos_ext*.so")
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        for pattern in patterns:
            if any(candidate.glob(pattern)):
                return candidate
    raise RuntimeError(
        "Could not find a build directory with nsos_ext.  Searched: "
        + ", ".join(str(c) for c in candidates)
    )


def load_nsos(build_dir: Path):
    build_dir = build_dir.resolve()
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    if os.name == "nt":
        add_windows_runtime_dirs(
            build_dir,
            parse_preferred_cuda_root(os.environ.get("NSOS_CUDA_ROOT")),
        )
    import nsos_ext as nsos  # type: ignore

    return nsos


def maybe_enable_hybrid_resume_cuda_safe_mode(args, profile: Dict, logger: RunLogger) -> None:
    if os.name != "nt":
        return
    if not args.resume_model:
        return
    if profile.get("profile_family") != "hybrid":
        return
    if args.device == "cpu":
        return

    enabled: List[str] = []
    if not os.environ.get("NSOS_CUDA_SYNC"):
        os.environ["NSOS_CUDA_SYNC"] = "1"
        enabled.append("NSOS_CUDA_SYNC=1")
    if not os.environ.get("CUDA_LAUNCH_BLOCKING"):
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        enabled.append("CUDA_LAUNCH_BLOCKING=1")
    if enabled:
        logger.log(
            "[device] enabling hybrid GPU resume safe mode on Windows: "
            + ", ".join(enabled)
        )


def ensure_bundle(
    repo_root: Path,
    bundle_dir: Path,
    seed: int,
    target_vocab: int,
    rebuild: bool,
    phase_sizes: Dict[str, Dict[str, int]] | None = None,
) -> Path:
    manifest_path = bundle_dir / "curriculum_manifest.json"
    tokenizer_path = bundle_dir / f"tokenizer_{target_vocab}.ox3"
    if rebuild or not manifest_path.exists():
        build_curriculum(repo_root, bundle_dir, seed=seed, phase_sizes=phase_sizes)
    if rebuild or not tokenizer_path.exists():
        build_tokenizer_bundle(bundle_dir, target_vocab=target_vocab)
    return tokenizer_path


def build_token_stream(tokenizer, rows: List[Dict], eos_token: str) -> List[int]:
    tokens: List[int] = []
    eos_tokens = tokenizer.encode(eos_token)
    eos_id = eos_tokens[0] if eos_tokens else None
    ordered_rows = sorted(
        rows,
        key=lambda row: row.get("id", "") or row.get("source", "") or row.get("kind", ""),
    )
    for row in ordered_rows:
        encoded = tokenizer.encode(row["text"])
        tokens.extend(encoded)
        if eos_id is not None and (not encoded or encoded[-1] != eos_id):
            tokens.append(eos_id)
    return tokens


def postprocess_generation(text: str) -> str:
    text = text.replace("<|endoftext|>", "").strip()
    if "\n" in text:
        text = text.split("\n", 1)[0].strip()
    return text


def safe_decode_tokens(tokenizer, token_ids: List[int]) -> str:
    if not token_ids:
        return ""
    try:
        return tokenizer.decode(token_ids)
    except UnicodeDecodeError:
        pieces: List[str] = []
        for token_id in token_ids:
            try:
                piece = tokenizer.decode([token_id])
            except UnicodeDecodeError:
                piece = ""
            if piece:
                pieces.append(piece)
            else:
                pieces.append(f"<0x{token_id:02x}>")
        return "".join(pieces)


def debug_eval_enabled() -> bool:
    value = os.environ.get("NSOS_DEBUG_EVAL", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def debug_eval_log(logger, message: str) -> None:
    if logger is None or not debug_eval_enabled():
        return
    logger.log(f"[eval-debug] {message}")


def build_start_token_blocklist(tokenizer, eos_token_id: int) -> set[int]:
    cache_key = (int(tokenizer.vocab_size), int(eos_token_id))
    cached = _START_BLOCKLIST_CACHE.get(cache_key)
    if cached is not None:
        return set(cached)

    blocked: set[int] = set()
    for token_id in range(int(tokenizer.vocab_size)):
        piece = safe_decode_tokens(tokenizer, [token_id])
        if piece.startswith("<|"):
            blocked.add(token_id)
    if eos_token_id >= 0:
        blocked.discard(eos_token_id)
    _START_BLOCKLIST_CACHE[cache_key] = set(blocked)
    return blocked


def extract_task_kind(prompt: str) -> str:
    match = re.match(r"<\|task:([^|>]+)\|>", prompt)
    return match.group(1) if match else "document"


def build_task_start_blocklist(tokenizer, eos_token_id: int, prompt: str) -> set[int]:
    task_kind = extract_task_kind(prompt)
    cache_key = (int(tokenizer.vocab_size), int(eos_token_id), task_kind)
    cached = _TASK_START_BLOCKLIST_CACHE.get(cache_key)
    if cached is not None:
        return set(cached)

    blocked = build_start_token_blocklist(tokenizer, eos_token_id)
    for token_id in range(int(tokenizer.vocab_size)):
        if token_id == eos_token_id:
            continue
        piece = safe_decode_tokens(tokenizer, [token_id])
        stripped = piece.strip()
        if not stripped:
            blocked.add(token_id)
            continue

        first_visible = next((ch for ch in stripped if not ch.isspace()), "")
        if task_kind not in {"math_small", "dsl"}:
            if not first_visible or not first_visible.isalnum():
                blocked.add(token_id)
                continue
        elif first_visible and not (first_visible.isalnum() or first_visible == "-"):
            blocked.add(token_id)
            continue

        if task_kind in NATURAL_LANGUAGE_TASKS and stripped.lower() in CONTAMINATED_START_TEXTS:
            blocked.add(token_id)
            continue

        if task_kind in LOWER_SINGLE_TOKEN_TASKS:
            if not re.fullmatch(r"[a-z][a-z0-9_-]*", stripped):
                blocked.add(token_id)
                continue
        elif task_kind in UPPER_LABEL_TASKS:
            if not re.fullmatch(r"[A-Z][A-Z0-9_-]*", stripped):
                blocked.add(token_id)
                continue
        elif task_kind in INTEGER_START_TASKS:
            if not re.fullmatch(r"-?[0-9]+", stripped):
                blocked.add(token_id)
                continue
        elif task_kind in BINARY_START_TASKS:
            if not re.fullmatch(r"[01]+", stripped):
                blocked.add(token_id)
    _TASK_START_BLOCKLIST_CACHE[cache_key] = set(blocked)
    return blocked


def build_supervised_tokens(tokenizer, row: Dict, eos_token_id: int):
    prompt_tokens = tokenizer.encode(
        f"<|task:{row['kind']}|>\nPrompt:\n{row['prompt']}\nAnswer:\n"
    )
    answer_tokens = tokenizer.encode(row["answer"])
    if eos_token_id >= 0:
        answer_tokens = list(answer_tokens) + [eos_token_id]
    return prompt_tokens, answer_tokens


def apply_generation_guards(logits_row, generated: List[int], eos_token_id: int,
                            blocked_token_ids: set[int], min_new_tokens: int,
                            repetition_penalty: float, no_repeat_ngram_size: int) -> None:
    if generated and repetition_penalty > 1.0:
        penalty = repetition_penalty
        for token_id in set(generated):
            if 0 <= token_id < len(logits_row):
                if logits_row[token_id] >= 0.0:
                    logits_row[token_id] /= penalty
                else:
                    logits_row[token_id] *= penalty

    if len(generated) + 1 >= no_repeat_ngram_size and no_repeat_ngram_size > 1:
        banned: set[int] = set()
        ngram = no_repeat_ngram_size
        if ngram == 2:
            last_token = generated[-1]
            for index in range(len(generated) - 1):
                if generated[index] == last_token:
                    banned.add(generated[index + 1])
        else:
            prefix = tuple(generated[-(ngram - 1):])
            for index in range(len(generated) - ngram + 1):
                if tuple(generated[index:index + ngram - 1]) == prefix:
                    banned.add(generated[index + ngram - 1])
        for token_id in banned:
            if 0 <= token_id < len(logits_row):
                logits_row[token_id] = -1e9

    if len(generated) < min_new_tokens and 0 <= eos_token_id < len(logits_row):
        logits_row[eos_token_id] = -1e9
        for token_id in blocked_token_ids:
            if 0 <= token_id < len(logits_row):
                logits_row[token_id] = -1e9


def compute_repetition_metrics(tokenizer, text: str) -> Dict[str, float]:
    token_ids = tokenizer.encode(text) if text else []
    if len(token_ids) <= 1:
        return {"probe_repeat_rate": 0.0, "probe_repeat_run": 0.0, "probe_repetition_penalty": 0.0}

    unique_ratio = len(set(token_ids)) / len(token_ids)
    repeat_rate = max(0.0, 1.0 - unique_ratio)

    longest_run = 1
    current_run = 1
    for index in range(1, len(token_ids)):
        if token_ids[index] == token_ids[index - 1]:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1
    repeat_run = longest_run / len(token_ids)

    if len(token_ids) >= 2:
        bigrams = [tuple(token_ids[index:index + 2]) for index in range(len(token_ids) - 1)]
        bigram_repeat = max(0.0, 1.0 - len(set(bigrams)) / len(bigrams))
    else:
        bigram_repeat = 0.0

    repetition_penalty = 0.45 * repeat_rate + 0.35 * repeat_run + 0.20 * bigram_repeat
    return {
        "probe_repeat_rate": repeat_rate,
        "probe_repeat_run": repeat_run,
        "probe_repetition_penalty": repetition_penalty,
    }


def compute_prefix_match_ratio(tokenizer, expected: str, prediction: str) -> float:
    expected_tokens = tokenizer.encode(expected.strip()) if expected else []
    prediction_tokens = tokenizer.encode(prediction.strip()) if prediction else []
    if not expected_tokens:
        return 0.0
    matched = 0
    for exp_token, pred_token in zip(expected_tokens, prediction_tokens):
        if exp_token != pred_token:
            break
        matched += 1
    return matched / max(len(expected_tokens), 1)


def greedy_generate(
    nsos,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    eos_token_id: int,
    min_new_tokens: int = 1,
    blocked_token_ids: set[int] | None = None,
    task_kind: str = "",
    logger=None,
    label: str = "generate",
) -> str:
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        return ""
    task_kind = task_kind or extract_task_kind(prompt)
    if blocked_token_ids is None:
        blocked_token_ids = build_task_start_blocklist(tokenizer, eos_token_id, prompt)
    repetition_penalty, no_repeat_ngram_size = generation_guard_profile(task_kind)
    generated: List[int] = []
    supports_streaming = model.supports_streaming_inference()
    debug_eval_log(
        logger,
        f"{label}:start task={task_kind} prompt_tok={len(token_ids)} max_new={max_new_tokens} streaming={int(supports_streaming)}",
    )
    model.reset_session()
    if supports_streaming:
        model.set_streaming_inference(True)
        logits = model.forward_ids(token_ids, None)
        debug_eval_log(logger, f"{label}:prompt_prefill={len(token_ids)}")
        for _ in range(max_new_tokens):
            host_logits = logits.cpu() if logits.device == nsos.Device.GPU else logits
            values = host_logits.numpy()
            last_row = values[-1].copy()
            apply_generation_guards(
                last_row,
                generated,
                eos_token_id,
                blocked_token_ids,
                min_new_tokens,
                repetition_penalty,
                no_repeat_ngram_size,
            )
            next_token = int(last_row.argmax())
            if next_token == eos_token_id:
                break
            generated.append(next_token)
            logits = model.forward_ids([next_token], None)
        model.set_streaming_inference(False)
        debug_eval_log(logger, f"{label}:done generated_tok={len(generated)}")
        return safe_decode_tokens(tokenizer, generated)

    for _ in range(max_new_tokens):
        model.reset_session()
        logits = model.forward_ids(token_ids + generated, None)
        host_logits = logits.cpu() if logits.device == nsos.Device.GPU else logits
        values = host_logits.numpy()
        last_row = values[-1].copy()
        apply_generation_guards(
            last_row,
            generated,
            eos_token_id,
            blocked_token_ids,
            min_new_tokens,
            repetition_penalty,
            no_repeat_ngram_size,
        )
        next_token = int(last_row.argmax())
        if next_token == eos_token_id:
            break
        generated.append(next_token)
    debug_eval_log(logger, f"{label}:done generated_tok={len(generated)}")
    return safe_decode_tokens(tokenizer, generated)


def evaluate_exact(nsos, model, tokenizer, rows: List[Dict], eos_token_id: int, max_new_tokens: int = 32,
                   logger=None, label: str = "exact") -> Dict[str, float]:
    total = 0
    correct = 0
    for row_index, row in enumerate(rows):
        if not row["answer"]:
            continue
        debug_eval_log(logger, f"{label}:row={row_index} kind={row.get('kind', '')}:start")
        prompt = f"<|task:{row['kind']}|>\nPrompt:\n{row['prompt']}\nAnswer:\n"
        answer_budget = len(tokenizer.encode(row["answer"])) + 4
        prediction = greedy_generate(
            nsos,
            model,
            tokenizer,
            prompt,
            max(max_new_tokens, answer_budget),
            eos_token_id,
            blocked_token_ids=build_start_token_blocklist(tokenizer, eos_token_id),
            task_kind="",
            logger=logger,
            label=f"{label}:row={row_index}:generate",
        )
        prediction = postprocess_generation(prediction)
        if prediction == row["answer"].strip():
            correct += 1
        total += 1
        debug_eval_log(logger, f"{label}:row={row_index}:done pred_len={len(prediction)}")
    accuracy = (correct / total) if total else 0.0
    return {"exact_total": total, "exact_correct": correct, "exact_accuracy": accuracy}


def evaluate_text_loss(nsos, model, tokenizer, rows: List[Dict], seq_len: int, max_windows: int = 24) -> Dict[str, float]:
    if max_windows <= 0:
        return {"heldout_loss": 0.0}
    token_stream = build_token_stream(tokenizer, rows, "<|endoftext|>")
    if len(token_stream) < seq_len + 1:
        return {"heldout_loss": 0.0}

    losses: List[float] = []
    windows = 0
    for start in range(0, len(token_stream) - seq_len - 1, seq_len):
        if windows >= max_windows:
            break
        model.reset_session()
        input_ids = token_stream[start : start + seq_len]
        target_ids = token_stream[start + 1 : start + seq_len + 1]
        logits = model.forward_ids(input_ids, None)
        host_logits = logits.cpu() if logits.device == nsos.Device.GPU else logits
        loss, _ = host_logits.cross_entropy(target_ids)
        losses.append(float(loss))
        windows += 1

    return {"heldout_loss": sum(losses) / len(losses) if losses else 0.0}


def cross_entropy_from_numpy(logits, targets: List[int]) -> float:
    if len(targets) == 0:
        return 0.0
    target_array = np.asarray(targets, dtype=np.int64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    log_denominator = np.log(exp_values.sum(axis=1, keepdims=True))
    log_probs = shifted - log_denominator
    row_indices = np.arange(target_array.shape[0])
    return float(-log_probs[row_indices, target_array].mean())


def evaluate_masked_supervised(nsos, model, tokenizer, rows: List[Dict], eos_token_id: int,
                               batch_size: int | None = None) -> Dict[str, float]:
    if not rows:
        return {
            "answer_loss": 0.0,
            "first_token_accuracy": 0.0,
            "teacher_token_accuracy": 0.0,
            "teacher_token_total": 0,
        }

    total_loss = 0.0
    first_hits = 0
    teacher_hits = 0
    teacher_total = 0
    sample_count = 0
    items: List[Dict[str, Any]] = []

    for row in rows:
        if not row.get("answer"):
            continue
        prompt_tokens, answer_tokens = build_supervised_tokens(tokenizer, row, eos_token_id)
        if not prompt_tokens or not answer_tokens:
            continue

        inputs = list(prompt_tokens)
        if len(answer_tokens) > 1:
            inputs.extend(answer_tokens[:-1])
        items.append({
            "inputs": inputs,
            "answer_tokens": answer_tokens,
            "start": len(prompt_tokens) - 1,
            "end": len(prompt_tokens) - 1 + len(answer_tokens),
        })

    effective_batch_size = max(int(batch_size or EVAL_RUNTIME_OPTIONS["masked_batch_size"]), 1)
    has_batch_forward = hasattr(model, "forward_ids_batch") and effective_batch_size > 1

    for offset in range(0, len(items), effective_batch_size):
        batch = items[offset : offset + effective_batch_size]
        model.reset_session()
        if has_batch_forward and len(batch) > 1:
            logits = model.forward_ids_batch([item["inputs"] for item in batch], None)
            host_logits = logits.cpu() if logits.device == nsos.Device.GPU else logits
            batch_values = host_logits.numpy()
            for batch_index, item in enumerate(batch):
                values = batch_values[batch_index, item["start"] : item["end"], :]
                answer_tokens = item["answer_tokens"]
                total_loss += cross_entropy_from_numpy(values, answer_tokens)
                sample_count += 1
                if len(answer_tokens) > 0 and int(values[0].argmax()) == int(answer_tokens[0]):
                    first_hits += 1
                for row_index, token in enumerate(answer_tokens):
                    teacher_total += 1
                    if int(values[row_index].argmax()) == int(token):
                        teacher_hits += 1
            continue

        for item in batch:
            model.reset_session()
            logits = model.forward_ids(item["inputs"], None)
            answer_logits = logits.slice(0, item["start"], item["end"])
            host_logits = answer_logits.cpu() if answer_logits.device == nsos.Device.GPU else answer_logits
            loss, _ = host_logits.cross_entropy(item["answer_tokens"])
            total_loss += float(loss)
            sample_count += 1

            values = host_logits.numpy()
            answer_tokens = item["answer_tokens"]
            if len(answer_tokens) > 0 and int(values[0].argmax()) == int(answer_tokens[0]):
                first_hits += 1

            for row_index, token in enumerate(answer_tokens):
                teacher_total += 1
                if int(values[row_index].argmax()) == int(token):
                    teacher_hits += 1

    sample_count = max(sample_count, 1)
    return {
        "answer_loss": total_loss / sample_count,
        "first_token_accuracy": first_hits / sample_count,
        "teacher_token_accuracy": (teacher_hits / teacher_total) if teacher_total else 0.0,
        "teacher_token_total": teacher_total,
    }


def evaluate_generation_speed(nsos, model, tokenizer, eos_token_id: int,
                              logger=None, label: str = "speed") -> Dict[str, float]:
    prompt = "Prompt:\nExplain why edge inference prefers compact weights.\nAnswer:\n"
    debug_eval_log(logger, f"{label}:start")
    started = time.perf_counter()
    generated = greedy_generate(
        nsos,
        model,
        tokenizer,
        prompt,
        24,
        eos_token_id,
        logger=logger,
        label=f"{label}:generate",
    )
    elapsed = max(time.perf_counter() - started, 1e-6)
    generated_tokens = len(tokenizer.encode(generated)) if generated else 0
    debug_eval_log(logger, f"{label}:done tokens={generated_tokens} elapsed={elapsed:.3f}s")
    return {
        "gen_tokens": generated_tokens,
        "gen_elapsed_s": elapsed,
        "gen_tokens_per_s": (generated_tokens / elapsed) if generated_tokens else 0.0,
    }


def save_run_summary(path: Path, summary: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def model_training_mode(model) -> bool | None:
    if hasattr(model, "training_mode"):
        return bool(model.training_mode())
    return None


def set_model_training_mode(model, enabled: bool) -> None:
    if hasattr(model, "set_training_mode"):
        model.set_training_mode(bool(enabled))


def restore_model_training_mode(model, previous: bool | None) -> None:
    if previous is not None:
        set_model_training_mode(model, previous)


def set_layer_audit_phase(layer_audit, phase: str, step: int = 0) -> None:
    if layer_audit is None:
        return
    layer_audit.set_phase(phase)
    layer_audit.set_step(int(step))


def layer_audit_summary_to_dict(summary) -> Dict[str, Any]:
    if summary is None:
        return {}
    return {
        "phase": str(summary.phase),
        "records": int(summary.records),
        "forward_records": int(summary.forward_records),
        "backward_records": int(summary.backward_records),
        "router_records": int(summary.router_records),
        "token_contexts": int(summary.token_contexts),
        "training_steps": int(summary.training_steps),
        "total_nan": int(summary.total_nan),
        "total_inf": int(summary.total_inf),
        "max_latency_ms": float(summary.max_latency_ms),
        "max_l2_norm": float(summary.max_l2_norm),
        "layers_seen": [int(layer) for layer in summary.layers_seen],
        "stored_records": int(getattr(summary, "stored_records", 0)),
        "dropped_records": int(getattr(summary, "dropped_records", 0)),
        "truncated_contexts": int(getattr(summary, "truncated_contexts", 0)),
        "router_entropy_count": int(getattr(summary, "router_entropy_count", 0)),
        "router_entropy_min": float(getattr(summary, "router_entropy_min", 0.0)),
        "router_entropy_max": float(getattr(summary, "router_entropy_max", 0.0)),
        "router_entropy_mean": float(getattr(summary, "router_entropy_mean", 0.0)),
        "router_num_experts_max": int(getattr(summary, "router_num_experts_max", 0)),
        "healthy": bool(summary.healthy()),
    }


def write_layer_audit_snapshot(layer_audit, audit_path: Path | None) -> None:
    if layer_audit is None or audit_path is None:
        return
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    layer_audit.write_json(str(audit_path))


def load_holdout_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        if row.get("prompt") and row.get("answer"):
            row = dict(row)
            if "text" not in row:
                row["text"] = (
                    f"<|task:{row.get('kind', 'holdout')}|>\n"
                    f"Prompt:\n{row['prompt']}\n"
                    f"Answer:\n{row['answer']}"
                )
            rows.append(row)
    return rows


def compute_masked_answer_score(metrics: Dict[str, float]) -> float:
    return (
        1.20 * float(metrics.get("first_token_accuracy", 0.0))
        + 1.80 * float(metrics.get("teacher_token_accuracy", 0.0))
        - 0.70 * float(metrics.get("answer_loss", 0.0))
    )


def evaluate_holdout_kind_breakdown(nsos, model, tokenizer, rows: List[Dict],
                                    eos_token_id: int, layer_audit) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        kind = str(row.get("kind") or row.get("category") or "unknown")
        grouped.setdefault(kind, []).append(row)

    if not grouped:
        return {}

    previous_training_mode = model_training_mode(model)
    previous_audit_enabled = bool(layer_audit.enabled()) if layer_audit is not None else False
    if layer_audit is not None:
        layer_audit.set_enabled(False)
    set_model_training_mode(model, False)
    try:
        breakdown: Dict[str, Dict[str, float]] = {}
        for kind in sorted(grouped):
            kind_rows = grouped[kind]
            metrics = evaluate_masked_supervised(
                nsos,
                model,
                tokenizer,
                kind_rows,
                eos_token_id,
                batch_size=EVAL_RUNTIME_OPTIONS["masked_batch_size"],
            )
            breakdown[kind] = {
                "rows": float(len(kind_rows)),
                "answer_loss": float(metrics.get("answer_loss", 0.0)),
                "first_token_accuracy": float(metrics.get("first_token_accuracy", 0.0)),
                "teacher_token_accuracy": float(metrics.get("teacher_token_accuracy", 0.0)),
                "teacher_token_total": float(metrics.get("teacher_token_total", 0.0)),
                "masked_score": compute_masked_answer_score(metrics),
            }
        return breakdown
    finally:
        restore_model_training_mode(model, previous_training_mode)
        if layer_audit is not None:
            layer_audit.set_enabled(previous_audit_enabled)


def aggregate_holdout_kind_breakdown(results: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    totals: Dict[str, Dict[str, float]] = {}
    for holdout_metrics in results.values():
        for kind, metrics in holdout_metrics.get("kind_breakdown", {}).items():
            bucket = totals.setdefault(
                kind,
                {
                    "rows": 0.0,
                    "answer_loss_sum": 0.0,
                    "first_sum": 0.0,
                    "teacher_hits_proxy": 0.0,
                    "teacher_token_total": 0.0,
                },
            )
            rows = float(metrics.get("rows", 0.0))
            teacher_total = float(metrics.get("teacher_token_total", 0.0))
            bucket["rows"] += rows
            bucket["answer_loss_sum"] += float(metrics.get("answer_loss", 0.0)) * rows
            bucket["first_sum"] += float(metrics.get("first_token_accuracy", 0.0)) * rows
            bucket["teacher_hits_proxy"] += (
                float(metrics.get("teacher_token_accuracy", 0.0)) * teacher_total
            )
            bucket["teacher_token_total"] += teacher_total

    aggregate: Dict[str, Dict[str, float]] = {}
    for kind, bucket in sorted(totals.items()):
        rows = max(float(bucket.get("rows", 0.0)), 1.0)
        teacher_total = float(bucket.get("teacher_token_total", 0.0))
        metrics = {
            "rows": float(bucket.get("rows", 0.0)),
            "answer_loss": float(bucket.get("answer_loss_sum", 0.0)) / rows,
            "first_token_accuracy": float(bucket.get("first_sum", 0.0)) / rows,
            "teacher_token_accuracy": (
                float(bucket.get("teacher_hits_proxy", 0.0)) / teacher_total
                if teacher_total > 0.0
                else 0.0
            ),
            "teacher_token_total": teacher_total,
        }
        metrics["masked_score"] = compute_masked_answer_score(metrics)
        aggregate[kind] = metrics
    return aggregate


def make_minipack_record(phase_name: str, kind: str, prompt: str, answer: str,
                         index: int) -> Dict[str, str]:
    record_id = hashlib.sha256(
        f"{phase_name}:{kind}:{prompt}:{answer}:{index}".encode("utf-8")
    ).hexdigest()[:16]
    return {
        "id": f"weak-kind-{record_id}",
        "phase": phase_name,
        "kind": kind,
        "prompt": prompt,
        "answer": answer,
        "source": "weak_kind_minipack",
        "text": (
            f"<|task:{kind}|>\n"
            f"Prompt:\n{prompt}\n"
            f"Answer:\n{answer}<|endoftext|>"
        ),
    }


def evaluate_binary_gate(op: str, a: int, b: int) -> int:
    if op == "AND":
        return a & b
    if op == "OR":
        return a | b
    if op == "XOR":
        return a ^ b
    if op == "NAND":
        return 1 - (a & b)
    if op == "NOR":
        return 1 - (a | b)
    if op == "XNOR":
        return 1 - (a ^ b)
    return 0


def generate_weak_kind_example(kind: str, rng: random.Random, index: int) -> tuple[str, str]:
    names = ["mila", "ivan", "lia", "nora", "otto", "ravi", "sara", "teo"]
    cities = ["belem", "curitiba", "natal", "coimbra", "evora", "madrid", "lima", "quito"]
    tools = ["nsos", "oxtamem", "bitnet", "jamba", "router", "audit", "pack", "kernel"]
    langs = ["pt", "en", "es"]

    if kind == "reverse":
        token = rng.choice(["edge9", "router42", "pack17", "audit5", "moe31"]) + rng.choice(["a", "b", "x"])
        return f"Reverse this token stream: {token}", token[::-1]
    if kind == "parity":
        bits = "".join(rng.choice("01") for _ in range(rng.randint(5, 9)))
        return (
            f"Parity bit for {bits}. Answer with 0 for even and 1 for odd.",
            str(bits.count("1") % 2),
        )
    if kind == "parity_label":
        bits = "".join(rng.choice("01") for _ in range(rng.randint(4, 8)))
        answer = "ODD" if bits.count("1") % 2 else "EVEN"
        return f"Parity for {bits}. Answer with EVEN or ODD.", answer
    if kind == "binary_add":
        lhs = rng.randint(2, 29)
        rhs = rng.randint(2, 29)
        return (
            f"Add the binary numbers {lhs:b} + {rhs:b}. Answer in binary.",
            f"{lhs + rhs:b}",
        )
    if kind == "compare":
        lhs = rng.randint(-25, 25)
        rhs = rng.randint(-25, 25)
        answer = ">" if lhs > rhs else "<" if lhs < rhs else "="
        return f"Compare {lhs} and {rhs}. Answer with <, > or =.", answer
    if kind == "compare_label":
        lhs = rng.randint(-25, 25)
        rhs = rng.randint(-25, 25)
        answer = "GT" if lhs > rhs else "LT" if lhs < rhs else "EQ"
        return f"Compare {lhs} and {rhs}. Answer with LT, GT, or EQ.", answer
    if kind == "circuit":
        op = rng.choice(["AND", "OR", "XOR", "NAND", "NOR", "XNOR"])
        a = rng.randint(0, 1)
        b = rng.randint(0, 1)
        return (
            f"Circuit solve: gate={op} A={a} B={b}. Answer with 0 or 1.",
            str(evaluate_binary_gate(op, a, b)),
        )
    if kind == "boolean_formula":
        op1 = rng.choice(["AND", "OR", "XOR"])
        op2 = rng.choice(["AND", "OR", "XOR"])
        a = rng.randint(0, 1)
        b = rng.randint(0, 1)
        c = rng.randint(0, 1)
        mid = evaluate_binary_gate(op1, a, b)
        value = evaluate_binary_gate(op2, mid, c)
        return f"Evaluate (({a} {op1} {b}) {op2} {c}). Answer with 0 or 1.", str(value)
    if kind == "code_output":
        a = rng.randint(2, 9)
        b = rng.randint(1, 8)
        c = rng.randint(2, 5)
        if index % 2 == 0:
            expr = f"({a} + {b}) * {c}"
            answer = str((a + b) * c)
        else:
            expr = f"({a} * {c}) - {b}"
            answer = str((a * c) - b)
        return f"What is the output of this Python snippet?\nprint({expr})", answer
    if kind == "dsl":
        start = rng.randint(1, 8)
        add = rng.randint(1, 8)
        mul = rng.randint(2, 4)
        sub = rng.randint(0, 8)
        answer = (start + add) * mul - sub
        return (
            "Execute this mini DSL and answer with the final integer: "
            f"SET {start} | ADD {add} | MUL {mul} | SUB {sub}",
            str(answer),
        )
    if kind == "kv_to_csv":
        name = rng.choice(["nsos", "bitnet", "oxtamem", "jamba"])
        lang = rng.choice(["en", "pt", "es"])
        mode = rng.choice(["edge", "api", "train", "audit"])
        return (
            "Normalize the key=value pairs into CSV ordered as name,lang,mode: "
            f"name={name} lang={lang} mode={mode}",
            f"{name},{lang},{mode}",
        )
    if kind == "convert_json":
        name = rng.choice(["nsos", "bitnet", "oxtamem", "jamba"])
        mode = rng.choice(["api", "edge", "audit", "train"])
        lang = rng.choice(["pt", "en", "es"])
        payload = {"name": name, "mode": mode, "lang": lang}
        return (
            f"Converta para JSON compacto: name={name} mode={mode} lang={lang}",
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
        )
    if kind == "logparse":
        level = rng.choice(["INFO", "WARN", "ERROR"])
        user = rng.choice(names)
        action = rng.choice(["train", "deploy", "audit", "reload"])
        status = rng.choice(["ok", "retry", "fail", "pass"])
        payload = {"level": level, "user": user, "action": action, "status": status}
        return (
            "Convert this structured log line to compact JSON with keys level,user,action,status: "
            f"[{level}] user={user} action={action} status={status}",
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
        )
    if kind == "translate":
        examples = [
            ("The router keeps experts balanced during training.", "O router mantem experts balanceados durante o treino."),
            ("Compact model packs reduce edge latency.", "Model packs compactos reduzem a latencia em edge."),
            ("The audit report catches unstable layers early.", "O relatorio de auditoria detecta camadas instaveis cedo."),
            ("Replay protects old skills during later phases.", "Replay protege habilidades antigas nas fases finais."),
        ]
        return rng.choice(examples)
    if kind == "summarize":
        examples = [
            (
                "Resuma em uma frase curta: Layer audits record shape, norm, latency, and router entropy for each phase.",
                "Resumo: Layer audits registram metricas por fase.",
            ),
            (
                "Resuma em uma frase curta: Balanced replay keeps algorithmic tasks visible during instruction training.",
                "Resumo: Replay balanceado preserva tarefas algoritmicas.",
            ),
            (
                "Resuma em uma frase curta: Holdout gates compare checkpoints and reject severe kind regressions.",
                "Resumo: Holdout gates rejeitam regressoes por kind.",
            ),
        ]
        return rng.choice(examples)
    if kind == "memory_recall":
        facts = {
            "name": rng.choice(names),
            "city": rng.choice(cities),
            "tool": rng.choice(tools),
            "lang": rng.choice(langs),
        }
        query_key = rng.choice(list(facts.keys()))
        prompt = "System: store the following session facts.\n"
        for key, value in facts.items():
            prompt += f"User: {key}={value}\nAssistant: memorized.\n"
        prompt += f"User: What is the {query_key}?\nAssistant:"
        return prompt, facts[query_key]

    return f"Echo the token: weak{index}", f"weak{index}"


def build_weak_kind_minipack(profile: Dict, phase_name: str, seed: int) -> List[Dict]:
    config = profile.get("weak_kind_minipack", {})
    if not config or not bool(config.get("enabled", False)):
        return []
    phase_counts = config.get("phase_counts", {})
    target_count = int(phase_counts.get(phase_name, 0))
    if target_count <= 0:
        return []
    kinds = [str(kind) for kind in config.get("kinds", [])]
    if not kinds:
        return []

    rng = random.Random(seed + sum(ord(ch) for ch in phase_name) + 515151)
    per_kind = max(1, math.ceil(target_count / len(kinds)))
    rows: List[Dict] = []
    index = 0
    for kind in kinds:
        for _ in range(per_kind):
            prompt, answer = generate_weak_kind_example(kind, rng, index)
            rows.append(make_minipack_record(phase_name, kind, prompt, answer, index))
            index += 1
    rng.shuffle(rows)
    return rows[:target_count]


def update_best_kind_scores(best_scores: Dict[str, float],
                            kind_breakdown: Dict[str, Dict[str, float]]) -> None:
    for kind, metrics in kind_breakdown.items():
        score = float(metrics.get("masked_score", 0.0))
        if kind not in best_scores or score > best_scores[kind]:
            best_scores[kind] = score


def compute_kind_regression_penalty(kind_breakdown: Dict[str, Dict[str, float]],
                                    best_scores: Dict[str, float],
                                    penalty_scale: float,
                                    regression_floor: float) -> Dict[str, object]:
    if penalty_scale <= 0.0 or not kind_breakdown or not best_scores:
        return {
            "penalty": 0.0,
            "weighted_regression": 0.0,
            "regressions": [],
        }

    weighted_regression = 0.0
    total_rows = 0.0
    regressions: List[Dict[str, float | str]] = []
    for kind, metrics in kind_breakdown.items():
        if kind not in best_scores:
            continue
        current_score = float(metrics.get("masked_score", 0.0))
        best_score = float(best_scores[kind])
        regression = best_score - current_score
        if regression <= regression_floor:
            continue
        rows = max(float(metrics.get("rows", 1.0)), 1.0)
        weighted_regression += regression * rows
        total_rows += rows
        regressions.append(
            {
                "kind": kind,
                "rows": rows,
                "best_score": best_score,
                "current_score": current_score,
                "regression": regression,
            }
        )

    if total_rows <= 0.0:
        return {
            "penalty": 0.0,
            "weighted_regression": 0.0,
            "regressions": [],
        }

    regressions.sort(key=lambda item: float(item["regression"]), reverse=True)
    weighted_average = weighted_regression / total_rows
    return {
        "penalty": weighted_average * penalty_scale,
        "weighted_regression": weighted_average,
        "regressions": regressions[:8],
    }


def evaluate_kind_regression_gate(kind_breakdown: Dict[str, Dict[str, float]],
                                  best_scores: Dict[str, float],
                                  gate_config: Dict[str, object]) -> Dict[str, object]:
    if not gate_config or not bool(gate_config.get("enabled", False)):
        return {
            "enabled": False,
            "passed": True,
            "failure_count": 0,
            "failures": [],
        }

    max_regression = float(gate_config.get("max_regression", 0.0))
    failures: List[Dict[str, float | str]] = []
    for kind, metrics in sorted(kind_breakdown.items()):
        if kind not in best_scores:
            continue
        current_score = float(metrics.get("masked_score", 0.0))
        best_score = float(best_scores[kind])
        regression = best_score - current_score
        if regression > max_regression:
            failures.append(
                {
                    "kind": kind,
                    "best_score": best_score,
                    "current_score": current_score,
                    "regression": regression,
                    "max_regression": max_regression,
                }
            )

    failures.sort(key=lambda item: float(item["regression"]), reverse=True)
    return {
        "enabled": True,
        "passed": not failures,
        "failure_count": len(failures),
        "max_regression": max_regression,
        "failures": failures,
    }


def evaluate_official_holdouts(nsos, model, tokenizer, holdout_files: List[Path],
                               eos_token_id: int, seq_len: int, eval_mode: str,
                               exact_samples: int, logger, layer_audit,
                               audit_phase_prefix: str = "holdout",
                               label_prefix: str = "holdout") -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    for holdout_path in holdout_files:
        rows = load_holdout_jsonl(holdout_path)
        label = holdout_path.stem
        if not rows:
            results[holdout_path.name] = {
                "path": str(holdout_path),
                "rows": 0,
                "missing": not holdout_path.exists(),
            }
            logger.log(f"[holdout] {holdout_path.name}: no rows")
            continue
        set_layer_audit_phase(layer_audit, f"{audit_phase_prefix}:{label}", 0)
        metrics = evaluate_phase(
            nsos,
            model,
            tokenizer,
            rows,
            eos_token_id,
            seq_len,
            eval_mode,
            exact_samples,
            logger=logger,
            label=f"{label_prefix}:{label}",
        )
        results[holdout_path.name] = {
            "path": str(holdout_path),
            "rows": len(rows),
            **metrics,
            "kind_breakdown": evaluate_holdout_kind_breakdown(
                nsos,
                model,
                tokenizer,
                rows,
                eos_token_id,
                layer_audit,
            ),
        }
        kind_breakdown = results[holdout_path.name]["kind_breakdown"]
        if kind_breakdown:
            weakest_kind, weakest_metrics = min(
                kind_breakdown.items(),
                key=lambda item: float(item[1].get("masked_score", 0.0)),
            )
            strongest_kind, strongest_metrics = max(
                kind_breakdown.items(),
                key=lambda item: float(item[1].get("masked_score", 0.0)),
            )
            logger.log(
                f"[holdout-kind] {holdout_path.name}: "
                f"best={strongest_kind}:{strongest_metrics['masked_score']:.4f} "
                f"worst={weakest_kind}:{weakest_metrics['masked_score']:.4f}"
            )
        logger.log(
            f"[holdout] {holdout_path.name}: rows={len(rows)} "
            f"answer_loss={metrics['answer_loss']:.4f} "
            f"first={metrics['first_token_accuracy']:.2f} "
            f"teacher={metrics['teacher_token_accuracy']:.2f} "
            f"exact={metrics['exact_correct']}/{metrics['exact_total']}"
        )
    return results


def collect_answer_logits(nsos, model, tokenizer, rows: List[Dict],
                          eos_token_id: int, limit: int) -> List[List[float]]:
    collected: List[List[float]] = []
    previous_training_mode = model_training_mode(model)
    set_model_training_mode(model, False)
    try:
        for row in rows[: max(1, limit)]:
            prompt_tokens, answer_tokens = build_supervised_tokens(tokenizer, row, eos_token_id)
            if not prompt_tokens or not answer_tokens:
                continue
            inputs = list(prompt_tokens)
            if len(answer_tokens) > 1:
                inputs.extend(answer_tokens[:-1])
            model.reset_session()
            logits = model.forward_ids(inputs, None)
            start = len(prompt_tokens) - 1
            end = start + len(answer_tokens)
            answer_logits = logits.slice(0, start, end)
            host_logits = answer_logits.cpu() if answer_logits.device == nsos.Device.GPU else answer_logits
            collected.append([float(value) for value in host_logits.numpy().reshape(-1).tolist()])
    finally:
        restore_model_training_mode(model, previous_training_mode)
    return collected


def compute_max_abs_drift(lhs: List[List[float]], rhs: List[List[float]]) -> float:
    max_drift = 0.0
    for lhs_values, rhs_values in zip(lhs, rhs):
        for lhs_value, rhs_value in zip(lhs_values, rhs_values):
            drift = abs(float(lhs_value) - float(rhs_value))
            if drift > max_drift:
                max_drift = drift
    return max_drift


def run_reload_probe(nsos, model, model_config, device, tokenizer, probe_rows: List[Dict],
                     eos_token_id: int, run_dir: Path, layer_audit, logger,
                     sample_limit: int = 4) -> Dict[str, Any]:
    if not probe_rows:
        return {"enabled": False, "reason": "no_probe_rows", "max_abs_drift": 0.0}

    probe_path = run_dir / "audit_reload_probe.bin"
    model.save(str(probe_path))

    set_layer_audit_phase(layer_audit, "reload_probe:pre", 0)
    pre_logits = collect_answer_logits(
        nsos,
        model,
        tokenizer,
        probe_rows,
        eos_token_id,
        sample_limit,
    )

    reloaded_model = nsos.JambaModel(model_config, device)
    reloaded_model.to(device)
    reloaded_model.load(str(probe_path), True)
    if layer_audit is not None:
        reloaded_model.set_audit_collector(layer_audit)

    set_layer_audit_phase(layer_audit, "reload_probe:post", 0)
    post_logits = collect_answer_logits(
        nsos,
        reloaded_model,
        tokenizer,
        probe_rows,
        eos_token_id,
        sample_limit,
    )
    drift = compute_max_abs_drift(pre_logits, post_logits)
    logger.log(
        f"[audit] reload_probe samples={min(len(pre_logits), len(post_logits))} "
        f"max_abs_drift={drift:.8f}"
    )
    return {
        "enabled": True,
        "path": str(probe_path),
        "samples": min(len(pre_logits), len(post_logits)),
        "max_abs_drift": drift,
    }


def train_rows_direct(trainer, tokenizer, rows: List[Dict], max_steps: int, callback,
                      seed: int, eos_token_id: int, batch_size: int) -> Dict[str, float]:
    if not rows or max_steps <= 0:
        return {}

    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    cursor = 0
    step = 0
    auxiliary_total: Dict[str, float] = {}
    while step < max_steps:
        if cursor >= len(order):
            rng.shuffle(order)
            cursor = 0

        prompt_batch: List[List[int]] = []
        answer_batch: List[List[int]] = []
        while cursor < len(order) or prompt_batch:
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            row = rows[order[cursor]]
            cursor += 1
            if not row.get("answer"):
                continue
            prompt_tokens, answer_tokens = build_supervised_tokens(tokenizer, row, eos_token_id)
            if not prompt_tokens or not answer_tokens:
                continue
            prompt_batch.append(prompt_tokens)
            answer_batch.append(answer_tokens)
            if len(prompt_batch) >= max(1, batch_size):
                break

        if not prompt_batch:
            break

        step += 1
        loss = trainer.train_supervised_batch(prompt_batch, answer_batch)
        if hasattr(trainer, "last_auxiliary_stats"):
            merge_auxiliary_metrics(
                auxiliary_total,
                auxiliary_stats_to_dict(trainer.last_auxiliary_stats),
            )
        if callback is not None:
            callback(step, float(loss))
    return finalize_auxiliary_metrics(auxiliary_total)


def sample_phase_replay_rows(
    phase_name: str,
    history_by_family: Dict[str, List[Dict]],
    ratio: float,
    current_count: int,
    rng: random.Random,
    profile_replay_config: Dict[str, Dict[str, object]] | None = None,
) -> List[Dict]:
    config = PHASE_REPLAY_CONFIG.get(phase_name, {"families": [], "ratio_scale": 0.0})
    if profile_replay_config and phase_name in profile_replay_config:
        config = profile_replay_config[phase_name]
    effective_ratio = float(ratio) * float(config.get("ratio_scale", 0.0))
    families = list(config.get("families", []))
    if effective_ratio <= 0.0 or current_count <= 0 or not families:
        return []

    replay_count = max(1, int(current_count * effective_ratio))
    picked: List[Dict] = []
    seen: set[str] = set()
    per_family = max(1, replay_count // max(len(families), 1))
    for family in families:
        family_rows = list(history_by_family.get(family, []))
        if not family_rows:
            continue
        take = min(len(family_rows), per_family)
        for row in rng.sample(family_rows, take) if take < len(family_rows) else family_rows:
            row_id = str(row.get("id", id(row)))
            if row_id in seen:
                continue
            seen.add(row_id)
            picked.append(row)

    pool: List[Dict] = []
    for family in families:
        pool.extend(history_by_family.get(family, []))
    if not pool:
        return []
    if len(picked) >= replay_count:
        rng.shuffle(picked)
        return picked[:replay_count]

    remaining = [row for row in pool if str(row.get("id", id(row))) not in seen]
    rng.shuffle(remaining)
    picked.extend(remaining[: max(0, replay_count - len(picked))])
    rng.shuffle(picked)
    return picked


def evaluate_generation_probe(nsos, model, tokenizer, rows: List[Dict], eos_token_id: int,
                              logger=None, label: str = "probe",
                              sample_count: int | None = None) -> Dict[str, float | str]:
    effective_samples = (
        EVAL_RUNTIME_OPTIONS["generation_probe_samples"]
        if sample_count is None
        else max(int(sample_count), 0)
    )
    probe_rows = [row for row in rows if row.get("answer")][: min(effective_samples, len(rows))]
    if not probe_rows:
        return {
            "probe_kind": "",
            "probe_expected": "",
            "probe_prediction": "",
            "probe_nonempty": 0.0,
            "probe_exact_match": 0.0,
            "probe_prefix_match_ratio": 0.0,
            "probe_repeat_rate": 0.0,
            "probe_repeat_run": 0.0,
            "probe_repetition_penalty": 0.0,
        }

    probe_predictions: List[str] = []
    probe_exact = 0.0
    probe_prefix = 0.0
    probe_nonempty = 0.0
    repetition_values = {
        "probe_repeat_rate": 0.0,
        "probe_repeat_run": 0.0,
        "probe_repetition_penalty": 0.0,
    }
    for probe_index, probe_row in enumerate(probe_rows):
        debug_eval_log(logger, f"{label}:row={probe_index} kind={probe_row.get('kind', '')}:start")
        prompt = f"<|task:{probe_row['kind']}|>\nPrompt:\n{probe_row['prompt']}\nAnswer:\n"
        answer_budget = len(tokenizer.encode(probe_row["answer"])) + 4
        prediction = greedy_generate(
            nsos,
            model,
            tokenizer,
            prompt,
            max(16, answer_budget),
            eos_token_id,
            blocked_token_ids=build_start_token_blocklist(tokenizer, eos_token_id),
            task_kind="",
            logger=logger,
            label=f"{label}:row={probe_index}:generate",
        )
        prediction = postprocess_generation(prediction)
        probe_predictions.append(prediction)
        probe_nonempty += 1.0 if prediction else 0.0
        probe_exact += 1.0 if prediction == probe_row["answer"].strip() else 0.0
        probe_prefix += compute_prefix_match_ratio(tokenizer, probe_row["answer"], prediction)
        metrics = compute_repetition_metrics(tokenizer, prediction)
        for key in repetition_values:
            repetition_values[key] += float(metrics[key])
        debug_eval_log(logger, f"{label}:row={probe_index}:done pred_len={len(prediction)}")

    denom = float(max(len(probe_rows), 1))
    first_row = probe_rows[0]
    return {
        "probe_kind": first_row["kind"],
        "probe_expected": first_row["answer"],
        "probe_prediction": probe_predictions[0] if probe_predictions else "",
        "probe_nonempty": probe_nonempty / denom,
        "probe_exact_match": probe_exact / denom,
        "probe_prefix_match_ratio": probe_prefix / denom,
        **{key: value / denom for key, value in repetition_values.items()},
    }


def evaluate_phase_impl(nsos, model, tokenizer, rows: List[Dict], eos_token_id: int,
                        seq_len: int, eval_mode: str, exact_samples: int,
                        logger=None, label: str = "phase_eval") -> Dict[str, float]:
    debug_eval_log(logger, f"{label}:masked:start rows={len(rows)}")
    masked_metrics = evaluate_masked_supervised(
        nsos,
        model,
        tokenizer,
        rows,
        eos_token_id,
        batch_size=EVAL_RUNTIME_OPTIONS["masked_batch_size"],
    )
    debug_eval_log(logger, f"{label}:masked:done")
    debug_eval_log(logger, f"{label}:text:start")
    text_metrics = evaluate_text_loss(
        nsos,
        model,
        tokenizer,
        rows,
        seq_len,
        max_windows=EVAL_RUNTIME_OPTIONS["text_loss_max_windows"],
    )
    debug_eval_log(logger, f"{label}:text:done")
    result = {**masked_metrics, **text_metrics}

    effective_exact_samples = (
        max(0, exact_samples)
        if eval_mode == "full"
        else max(0, EVAL_RUNTIME_OPTIONS["fast_exact_samples"])
    )
    exact_rows = [row for row in rows if row.get("answer")][:effective_exact_samples]
    if exact_rows:
        debug_eval_log(logger, f"{label}:exact:start rows={len(exact_rows)}")
        result.update(
            evaluate_exact(
                nsos,
                model,
                tokenizer,
                exact_rows,
                eos_token_id,
                logger=logger,
                label=f"{label}:exact",
            )
        )
        debug_eval_log(logger, f"{label}:exact:done")
    else:
        result.update({"exact_total": 0, "exact_correct": 0, "exact_accuracy": 0.0})

    if EVAL_RUNTIME_OPTIONS["generation_probe_samples"] > 0:
        debug_eval_log(logger, f"{label}:probe:start")
        result.update(
            evaluate_generation_probe(
                nsos,
                model,
                tokenizer,
                rows,
                eos_token_id,
                logger=logger,
                label=f"{label}:probe",
                sample_count=EVAL_RUNTIME_OPTIONS["generation_probe_samples"],
            )
        )
        debug_eval_log(logger, f"{label}:probe:done")
    else:
        result.update({
            "probe_kind": "",
            "probe_expected": "",
            "probe_prediction": "",
            "probe_nonempty": 0.0,
            "probe_exact_match": 0.0,
            "probe_prefix_match_ratio": 0.0,
            "probe_repeat_rate": 0.0,
            "probe_repeat_run": 0.0,
            "probe_repetition_penalty": 0.0,
        })

    if eval_mode == "full":
        result.update(
            evaluate_generation_speed(
                nsos,
                model,
                tokenizer,
                eos_token_id,
                logger=logger,
                label=f"{label}:speed",
            )
        )
    else:
        result.update({"gen_tokens": 0, "gen_elapsed_s": 0.0, "gen_tokens_per_s": 0.0})
    debug_eval_log(logger, f"{label}:done")
    return result


def evaluate_phase(nsos, model, tokenizer, rows: List[Dict], eos_token_id: int,
                   seq_len: int, eval_mode: str, exact_samples: int,
                   logger=None, label: str = "phase_eval") -> Dict[str, float]:
    previous_training_mode = model_training_mode(model)
    set_model_training_mode(model, False)
    try:
        return evaluate_phase_impl(
            nsos,
            model,
            tokenizer,
            rows,
            eos_token_id,
            seq_len,
            eval_mode,
            exact_samples,
            logger=logger,
            label=label,
        )
    finally:
        restore_model_training_mode(model, previous_training_mode)


def build_global_suite(bundle_dir: Path, samples_per_phase: int) -> Dict[str, List[Dict]]:
    suite: Dict[str, List[Dict]] = {}
    for phase_name in resolve_phase_order(None):
        suite[phase_name] = curriculum_texts_for_phase(bundle_dir, phase_name, "eval")[: max(1, samples_per_phase)]
    return suite


def evaluate_global_suite(
    nsos,
    model,
    tokenizer,
    suite_rows: Dict[str, List[Dict]],
    eos_token_id: int,
    seq_len: int,
    exact_samples: int,
    logger=None,
    label_prefix: str = "global",
) -> tuple[float, Dict[str, Dict[str, float]]]:
    total_score = 0.0
    phase_results: Dict[str, Dict[str, float]] = {}
    for phase_name, rows in suite_rows.items():
        metrics = evaluate_phase(
            nsos,
            model,
            tokenizer,
            rows,
            eos_token_id,
            seq_len,
            "full",
            max(exact_samples, 2),
            logger=logger,
            label=f"{label_prefix}:{phase_name}",
        )
        phase_results[phase_name] = metrics
        total_score += compute_phase_score(metrics)
    return total_score, phase_results


def compute_phase_score(metrics: Dict[str, float]) -> float:
    teacher_total = int(metrics.get("teacher_token_total", 0))
    if teacher_total <= 0:
        return (
            -1.15 * float(metrics.get("heldout_loss", 0.0))
            + 0.45 * float(metrics.get("probe_nonempty", 0.0))
            + 1.20 * float(metrics.get("probe_exact_match", 0.0))
            + 1.00 * float(metrics.get("probe_prefix_match_ratio", 0.0))
            - 0.90 * float(metrics.get("probe_repetition_penalty", 0.0))
        )
    return (
        1.20 * float(metrics.get("first_token_accuracy", 0.0))
        + 1.80 * float(metrics.get("teacher_token_accuracy", 0.0))
        + 1.55 * float(metrics.get("exact_accuracy", 0.0))
        + 1.25 * float(metrics.get("probe_exact_match", 0.0))
        + 1.15 * float(metrics.get("probe_prefix_match_ratio", 0.0))
        + 0.25 * float(metrics.get("probe_nonempty", 0.0))
        - 0.70 * float(metrics.get("answer_loss", 0.0))
        - 0.55 * float(metrics.get("heldout_loss", 0.0))
        - 1.10 * float(metrics.get("probe_repetition_penalty", 0.0))
    )


def compute_official_holdout_score(results: Dict[str, Dict]) -> Dict[str, float]:
    weighted: Dict[str, float] = {
        "answer_loss": 0.0,
        "heldout_loss": 0.0,
        "first_token_accuracy": 0.0,
        "teacher_token_accuracy": 0.0,
        "exact_accuracy": 0.0,
        "probe_exact_match": 0.0,
        "probe_prefix_match_ratio": 0.0,
        "probe_repetition_penalty": 0.0,
    }
    total_rows = 0
    weighted_score = 0.0
    for metrics in results.values():
        rows = int(metrics.get("rows", 0))
        if rows <= 0:
            continue
        total_rows += rows
        weighted_score += compute_phase_score(metrics) * rows
        for key in weighted:
            weighted[key] += float(metrics.get(key, 0.0)) * rows

    if total_rows <= 0:
        return {
            "score": float("-inf"),
            "rows": 0.0,
            **{key: 0.0 for key in weighted},
        }

    aggregate = {key: value / total_rows for key, value in weighted.items()}
    aggregate["score"] = weighted_score / total_rows
    aggregate["rows"] = float(total_rows)
    return aggregate


def main() -> int:
    # ── Boot heartbeats ───────────────────────────────────────────────
    # The startup path between `parse_args()` and the first
    # `logger.log(...)` involves: importing nsos_ext (CUDA libs init,
    # 30-90 s on first run), tokenizer load, GPU probe, and JambaModel
    # construction (allocates GPU memory + initializes weights).  When
    # this is launched as a subprocess inside Jupyter/Colab, those
    # heavy steps run silent unless we explicitly flush print() calls
    # with the `[boot]` prefix so the user can tell WHICH step is slow.
    # Without these heartbeats, "9 minutes silent" was indistinguishable
    # from "hung in CUDA init" vs "hung in JambaModel ctor" vs "running
    # fine but output buffered".
    print("[boot] train_curriculum.py main() entered", flush=True)
    args = parse_args()
    print(f"[boot] parsed args; profile={args.profile} device={args.device} bundle_dir={args.bundle_dir}", flush=True)
    EVAL_RUNTIME_OPTIONS.update({
        "masked_batch_size": max(int(args.eval_masked_batch_size), 1),
        "generation_probe_samples": max(int(args.generation_probe_samples), 0),
        "fast_exact_samples": max(int(args.fast_exact_samples), 0),
        "text_loss_max_windows": max(int(args.text_loss_max_windows), 0),
    })
    canonical_profile, profile = resolve_profile(args.profile)
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    layer_audit_path = args.layer_audit_out or (run_dir / "layer_audit.json")
    layer_audit_summary_path = args.audit_summary_out or (run_dir / "layer_audit_summary.json")
    layer_audit = None
    session_log_path = args.session_log or (run_dir / "session.log")
    logger = RunLogger(args.progress_mode, session_log_path)
    print(f"[boot] RunLogger ready; session.log -> {session_log_path}", flush=True)

    try:
        maybe_enable_hybrid_resume_cuda_safe_mode(args, profile, logger)
        build_dir = detect_build_dir(args.build_dir)
        print(f"[boot] build_dir resolved -> {build_dir}", flush=True)
        print(f"[boot] importing nsos_ext (CUDA libs init can take 30-90 s on first import)...", flush=True)
        nsos = load_nsos(build_dir)
        print(f"[boot] nsos_ext imported OK", flush=True)

        print(f"[boot] ensuring bundle at {args.bundle_dir} (rebuild={args.rebuild_curriculum})...", flush=True)
        tokenizer_path = ensure_bundle(
            args.repo_root,
            args.bundle_dir,
            seed=args.seed,
            target_vocab=profile["target_vocab"],
            rebuild=args.rebuild_curriculum,
            phase_sizes=profile.get("phase_sizes"),
        )
        print(f"[boot] bundle OK -> tokenizer artifact at {tokenizer_path}", flush=True)

        tokenizer = nsos.Tokenizer()
        tokenizer.load(str(tokenizer_path))
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
        tokenizer.save_pack(str(run_dir / "tokenizer.nsos"))
        eos_token_id = tokenizer.encode("<|endoftext|>")[0]
        print(f"[boot] tokenizer loaded (vocab={tokenizer.vocab_size}); tokenizer.nsos written", flush=True)

        if args.device == "gpu":
            device = nsos.Device.GPU
        elif args.device == "cpu":
            device = nsos.Device.CPU
        else:
            device = nsos.Device.GPU if os.name == "nt" else nsos.Device.CPU

        if device == nsos.Device.GPU and hasattr(nsos, "fast_gpu_supported"):
            print("[boot] probing fast_gpu_supported()...", flush=True)
            if not nsos.fast_gpu_supported():
                logger.log(
                    "[device] Fast GPU path unavailable on this CUDA/toolkit/GPU combination; "
                    "falling back to CPU for correctness and throughput."
                )
                device = nsos.Device.CPU
            else:
                print("[boot] fast_gpu_supported() = True", flush=True)

        print("[boot] building model_config + effective schedule...", flush=True)
        model_config = build_model_config(nsos, profile, tokenizer.vocab_size, device)
        model_config_dict = model_config_to_dict(model_config)
        effective_schedule = build_effective_schedule(profile, model_config_dict)
        (run_dir / "effective_model_config.json").write_text(
            json.dumps(model_config_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "effective_schedule.json").write_text(
            json.dumps(effective_schedule, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "support_matrix.json").write_text(
            json.dumps(OFFICIAL_SUPPORT_MATRIX, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("[boot] effective_model_config.json + effective_schedule.json + support_matrix.json written", flush=True)
        logger.log(
            f"[profile] requested={args.profile} canonical={canonical_profile} "
            f"family={effective_schedule['profile_family']} dominant={effective_schedule['dominant_stack']}"
        )
        logger.log(f"[profile] scope={effective_schedule['validation_scope']}")

        device_label = "GPU" if device == nsos.Device.GPU else "CPU"
        print(
            f"[boot] constructing JambaModel ({profile['layers']} layers, d_model={profile['d_model']}, "
            f"vocab={tokenizer.vocab_size}, device={device_label}) — allocates device memory + initializes weights, "
            f"this is the canonical 'silent' window on first run...",
            flush=True,
        )
        model = nsos.JambaModel(model_config, device)
        print("[boot] JambaModel constructed; moving to device...", flush=True)
        model.to(device)
        print(f"[boot] model on {device_label}", flush=True)
        # OXTA-CRIT Lei 2: prova de runtime do espectro de timescales do Mamba.
        # Linha definitiva para saber se NSOS_MAMBA_A_LOGSPACED chegou ao .so em
        # uso — spread ~0 = A=ones degenerado (fix INATIVO); spread >= ~2 décadas
        # = log-espaçado (fix ATIVO).  Lê os parâmetros do modelo vivo, então não
        # há como um binário stale mentir aqui.
        try:
            import numpy as _np
            _taus = []
            for _p in model.parameters():
                if _p.name.endswith("mamba.A"):
                    _a = _np.abs(_np.asarray(_p.data.numpy(), dtype=_np.float64))
                    _taus.append(1.0 / _np.maximum(_a, 1e-3))
            if _taus:
                _t = _np.concatenate(_taus)
                _spread = float(_np.log10(_t.max() / max(float(_t.min()), 1e-9)))
                _verdict = "LOGSPACED (fix ATIVO)" if _spread > 0.5 else "DEGENERADO A=ones (fix INATIVO)"
                print(f"[boot] mamba A spectrum: tau[min={_t.min():.1f} med={float(_np.median(_t)):.1f} "
                      f"max={_t.max():.1f}] spread_log10={_spread:.2f} -> {_verdict}", flush=True)
        except Exception as _exc:  # nunca derruba o boot por causa do probe
            print(f"[boot] mamba A spectrum probe indisponivel: {_exc}", flush=True)
        audit_min_layer_coverage = (
            int(args.audit_min_layer_coverage)
            if int(args.audit_min_layer_coverage) > 0
            else int(model_config.num_layers)
        )
        audit_thresholds = {
            "require_zero_nan_inf": True,
            "min_layer_coverage": audit_min_layer_coverage,
            "router_entropy_min": float(args.audit_router_entropy_min),
            "router_entropy_max": float(args.audit_router_entropy_max),
            "max_reload_drift": float(args.audit_max_reload_drift),
        }
        if args.enable_layer_audit:
            if not hasattr(nsos, "LayerAuditCollector"):
                raise RuntimeError("nsos_ext does not expose LayerAuditCollector; rebuild nsos_ext first.")
            layer_audit = nsos.LayerAuditCollector()
            layer_audit.begin_run(f"{canonical_profile}:seed{args.seed}")
            layer_audit.set_storage_policy(
                bool(args.audit_summary_only),
                max(int(args.audit_record_sample_rate), 1),
                max(int(args.audit_max_records_per_phase), 0),
                bool(args.audit_store_token_contexts),
            )
            layer_audit.set_enabled(True)
            model.set_audit_collector(layer_audit)
            logger.log(f"[audit] enabled path={layer_audit_path}")
        if args.resume_model is not None and args.resume_model.exists():
            try:
                model.load(str(args.resume_model), True)
                logger.log(f"[resume] strict checkpoint load ok: {args.resume_model}")
            except RuntimeError as strict_error:
                logger.log(
                    f"[resume] strict checkpoint load failed: {strict_error}; retrying partial load"
                )
                model.load(str(args.resume_model), False)
                logger.log(f"[resume] partial checkpoint load ok: {args.resume_model}")
        print(f"[boot] constructing Trainer (lr={profile['lr']}, warmup={profile['warmup_steps']})...", flush=True)
        trainer = nsos.Trainer(model, profile["lr"])
        trainer.weight_decay = profile["weight_decay"]
        trainer.max_grad_norm = profile["max_grad_norm"]
        trainer.warmup_steps = profile["warmup_steps"]
        trainer.min_learning_rate_scale = profile["min_lr_scale"]
        trainer.first_token_loss_scale = profile.get("first_token_loss_scale", 2.5)
        trainer.eos_loss_scale = profile.get("eos_loss_scale", 0.35)
        trainer.moe_aux_loss_scale = profile.get("moe_aux_loss_scale", trainer.moe_aux_loss_scale)
        trainer.repetition_unlikelihood_scale = phase_repetition_scale(
            profile,
            "phase1_algorithms",
        )
        trainer.eos_token_id = eos_token_id
        trainer.total_training_steps = (
            sum(args.override_phase_steps or profile["phase_steps"][phase] for phase in PHASE_ORDER)
            + int(profile.get("instruction_polish_steps", 0))
            + int(profile.get("weak_kind_rehearsal_steps", 0))
            + max(args.final_consolidation_steps, 0)
        )
        qat_cfg = profile.get("qat", {})
        qat_enabled = bool(qat_cfg.get("enabled", False)) and not args.disable_qat
        scheduler = nsos.TrainPhaseScheduler()
        if qat_enabled:
            scheduler.progressive_qat_enabled = True
            scheduler.semantic_warmup_steps = int(qat_cfg.get("semantic_warmup_steps", 0))
            scheduler.qat_start_step = int(qat_cfg.get("qat_start_step", scheduler.semantic_warmup_steps))
            scheduler.quantized_precision_bits = int(qat_cfg.get("quantized_precision_bits", 2))
            scheduler.ternary_regularization = float(qat_cfg.get("ternary_regularization", 0.0))
        trainer.configure_progressive_qat(scheduler)
        print(
            f"[boot] Trainer configured (total_training_steps={trainer.total_training_steps}, "
            f"qat={'on' if qat_enabled else 'off'}); preparing summary + entering phase loop next.",
            flush=True,
        )

        summary = {
            "profile": canonical_profile,
            "requested_profile": args.profile,
            "legacy_alias_used": bool(profile.get("legacy_alias_used", False)),
            "build_dir": str(build_dir),
            "bundle_dir": str(args.bundle_dir),
            "run_dir": str(run_dir),
            "device": "gpu" if device == nsos.Device.GPU else "cpu",
            "model": {
                "layers": profile["layers"],
                "d_model": profile["d_model"],
                "vocab_size": tokenizer.vocab_size,
                "target_vocab": profile["target_vocab"],
            },
            "model_config": model_config_dict,
            "effective_schedule": effective_schedule,
            "support_matrix": dict(OFFICIAL_SUPPORT_MATRIX),
            "optimizer": {
                "lr": profile["lr"],
                "weight_decay": profile["weight_decay"],
                "max_grad_norm": profile["max_grad_norm"],
                "warmup_steps": profile["warmup_steps"],
                "min_lr_scale": profile["min_lr_scale"],
                "first_token_loss_scale": trainer.first_token_loss_scale,
                "eos_loss_scale": trainer.eos_loss_scale,
                "moe_aux_loss_scale": trainer.moe_aux_loss_scale,
                "kind_regression_penalty_scale": float(
                    profile.get("kind_regression_penalty_scale", 0.0)
                ),
                "kind_regression_floor": float(profile.get("kind_regression_floor", 0.0)),
                "kind_regression_gate": deepcopy(profile.get("kind_regression_gate", {})),
                "weak_kind_minipack": deepcopy(profile.get("weak_kind_minipack", {})),
                "weak_kind_rehearsal_steps": int(profile.get("weak_kind_rehearsal_steps", 0)),
                "weak_kind_rehearsal_lr_scale": float(
                    profile.get("weak_kind_rehearsal_lr_scale", 1.0)
                ),
                "phase_replay_config": deepcopy(profile.get("phase_replay_config", {})),
                "repetition_unlikelihood_scale": float(profile.get("repetition_unlikelihood_scale", 0.0)),
                "phase_repetition_unlikelihood_scale": dict(
                    profile.get("phase_repetition_unlikelihood_scale", {})
                ),
                "instruction_polish_steps": int(profile.get("instruction_polish_steps", 0)),
                "instruction_polish_lr_scale": float(profile.get("instruction_polish_lr_scale", 1.0)),
                "batch_size": profile["batch_size"],
                "progressive_qat": {
                    "requested_disable_qat": bool(args.disable_qat),
                    "profile_enabled": bool(qat_cfg.get("enabled", False)),
                    "enabled": bool(getattr(trainer.phase_scheduler, "progressive_qat_enabled", False)),
                    "semantic_warmup_steps": int(getattr(trainer.phase_scheduler, "semantic_warmup_steps", 0)),
                    "qat_start_step": int(getattr(trainer.phase_scheduler, "qat_start_step", 0)),
                    "quantized_precision_bits": int(getattr(trainer.phase_scheduler, "quantized_precision_bits", 0)),
                    "ternary_regularization": float(getattr(trainer.phase_scheduler, "ternary_regularization", 0.0)),
                },
                "auxiliary_stack_profile": deepcopy(profile.get("auxiliary_stack", {})),
            },
            "resume_model": str(args.resume_model) if args.resume_model else "",
            "phase_eval_mode": args.phase_eval_mode,
            "phase_eval_samples": args.phase_eval_samples,
            "phase_exact_samples": args.phase_exact_samples,
            "fast_exact_samples": args.fast_exact_samples,
            "generation_probe_samples": args.generation_probe_samples,
            "eval_masked_batch_size": args.eval_masked_batch_size,
            "text_loss_max_windows": args.text_loss_max_windows,
            "global_suite_samples_per_phase": args.global_suite_samples_per_phase,
            "global_suite_exact_samples": args.global_suite_exact_samples,
            "replay_ratio": args.replay_ratio,
            "final_consolidation_steps": args.final_consolidation_steps,
            "phase_best_eval_every_steps": args.phase_best_eval_every_steps,
            "log_ema_beta": args.log_ema_beta,
            "progress_mode": logger.mode,
            "official_holdout_files": [str(path) for path in args.holdout_files],
            "holdout_eval_mode": args.holdout_eval_mode,
            "holdout_exact_samples": args.holdout_exact_samples,
            "holdout_every_phases": args.holdout_every_phases,
            "select_final_by_holdout": bool(args.select_final_by_holdout),
            "total_training_steps": trainer.total_training_steps,
            "generation_guard": {
                "repetition_penalty": GENERATION_REPETITION_PENALTY,
                "no_repeat_ngram_size": GENERATION_NO_REPEAT_NGRAM,
            },
            "layer_audit": {
                "enabled": bool(args.enable_layer_audit),
                "path": str(layer_audit_path),
                "summary_path": str(layer_audit_summary_path),
                "thresholds": audit_thresholds,
                "storage_policy": {
                    "summary_only": bool(args.audit_summary_only),
                    "record_sample_rate": max(int(args.audit_record_sample_rate), 1),
                    "max_records_per_phase": max(int(args.audit_max_records_per_phase), 0),
                    "store_token_contexts": bool(args.audit_store_token_contexts),
                },
                "phase_snapshots": [],
            },
            "official_holdouts": {},
            "official_holdout_kind_breakdown": {},
            "capacity_champion": {},
            "capacity_holdout_curve": [],
            "capacity_best_kind_scores": {},
            "capacity_kind_regression_gate": {
                "enabled": bool(profile.get("kind_regression_gate", {}).get("enabled", False)),
                "passed": True,
                "failure_count": 0,
                "failures": [],
            },
            "global_champion": {},
            "research_champion": {},
            "release_candidate": {},
            "release_gate": {
                "thresholds": deepcopy(RELEASE_GATE_THRESHOLDS),
                "passed": False,
                "failure_count": 0,
                "failures": [],
            },
            "phases": [],
        }

        def save_checkpoint_artifact(name: str) -> None:
            model.save(str(run_dir / f"{name}.bin"))
            model.save_edge_linear_pack(str(run_dir / f"{name}.edge.nsos"))
            shutil.copy2(run_dir / "tokenizer.nsos", run_dir / f"{name}.tokenizer.nsos")

        def consider_global_champion(source_name: str, source_phase: str) -> float:
            nonlocal best_global_score, best_release_score
            debug_eval_log(logger, f"global:start source={source_name}")
            set_layer_audit_phase(layer_audit, f"{source_phase}:global", 0)
            global_score, global_metrics = evaluate_global_suite(
                nsos,
                model,
                tokenizer,
                global_suite,
                eos_token_id,
                profile["seq_len"],
                args.global_suite_exact_samples,
                logger=logger,
                label_prefix=f"global:{source_phase}:{source_name}",
            )
            debug_eval_log(logger, f"global:done source={source_name} score={global_score:.4f}")
            release_gate = evaluate_release_gate(global_metrics)
            release_score = compute_release_candidate_score(global_metrics)
            logger.log(f"[global] source={source_name} score={global_score:.4f}")
            if global_score > best_global_score:
                best_global_score = global_score
                save_checkpoint_artifact("champion_global")
                summary["global_champion"] = {
                    "score": global_score,
                    "source": source_name,
                    "source_phase": source_phase,
                    "suite_samples_per_phase": args.global_suite_samples_per_phase,
                    "suite_exact_samples": args.global_suite_exact_samples,
                    "phase_metrics": global_metrics,
                }
                summary["research_champion"] = dict(summary["global_champion"])
                logger.log(f"[global] champion updated -> {source_name}")
            if release_gate["passed"] and release_score > best_release_score:
                best_release_score = release_score
                save_checkpoint_artifact("release_candidate")
                summary["release_candidate"] = {
                    "score": release_score,
                    "source": source_name,
                    "source_phase": source_phase,
                    "suite_samples_per_phase": args.global_suite_samples_per_phase,
                    "suite_exact_samples": args.global_suite_exact_samples,
                    "phase_metrics": global_metrics,
                }
                logger.log(f"[release] candidate updated -> {source_name}")
            summary["release_gate"] = {
                **release_gate,
                "thresholds": deepcopy(RELEASE_GATE_THRESHOLDS),
                "last_source": source_name,
                "last_score": release_score,
            }
            return global_score

        def consider_capacity_champion(source_name: str, source_phase: str) -> float:
            nonlocal best_capacity_score
            if int(args.holdout_every_phases) <= 0:
                return float("nan")
            debug_eval_log(logger, f"capacity_holdout:start source={source_name}")
            holdout_results = evaluate_official_holdouts(
                nsos,
                model,
                tokenizer,
                list(args.holdout_files or []),
                eos_token_id,
                profile["seq_len"],
                args.holdout_eval_mode,
                args.holdout_exact_samples,
                logger,
                layer_audit,
                audit_phase_prefix=f"{source_phase}:holdout",
                label_prefix=f"capacity_holdout:{source_phase}",
            )
            aggregate = compute_official_holdout_score(holdout_results)
            kind_breakdown = aggregate_holdout_kind_breakdown(holdout_results)
            raw_score = float(aggregate["score"])
            regression = compute_kind_regression_penalty(
                kind_breakdown,
                best_capacity_kind_scores,
                float(profile.get("kind_regression_penalty_scale", 0.0)),
                float(profile.get("kind_regression_floor", 0.0)),
            )
            candidate_kind_gate = evaluate_kind_regression_gate(
                kind_breakdown,
                best_capacity_kind_scores,
                profile.get("kind_regression_gate", {}),
            )
            candidate_kind_gate_passed = bool(candidate_kind_gate.get("passed", True))
            score = raw_score - float(regression["penalty"])
            record = {
                "source": source_name,
                "source_phase": source_phase,
                "score": score,
                "raw_score": raw_score,
                "kind_regression_penalty": float(regression["penalty"]),
                "kind_weighted_regression": float(regression["weighted_regression"]),
                "kind_regressions": regression["regressions"],
                "kind_gate": candidate_kind_gate,
                "aggregate": aggregate,
                "kind_breakdown": kind_breakdown,
                "holdouts": holdout_results,
            }
            summary["capacity_holdout_curve"].append(record)
            logger.log(
                f"[capacity] source={source_name} score={score:.4f} "
                f"raw={raw_score:.4f} penalty={float(regression['penalty']):.4f} "
                f"answer_loss={aggregate['answer_loss']:.4f} "
                f"teacher={aggregate['teacher_token_accuracy']:.2f}"
            )
            if candidate_kind_gate_passed and score > best_capacity_score:
                best_capacity_score = score
                save_checkpoint_artifact("champion_holdout")
                summary["capacity_champion"] = record
                logger.log(f"[capacity] champion updated -> {source_name}")
            elif not candidate_kind_gate_passed:
                logger.log(
                    f"[capacity] rejected by kind gate -> {source_name} "
                    f"failures={candidate_kind_gate.get('failure_count', 0)}"
                )
            if candidate_kind_gate_passed:
                update_best_kind_scores(best_capacity_kind_scores, kind_breakdown)
            debug_eval_log(logger, f"capacity_holdout:done source={source_name} score={score:.4f}")
            return score

        # AUDIT #8 (2026-05-16): write metrics to a LOCAL fast file
        # first and copy to the run_dir (which may be on Drive fuse,
        # ~5-10ms per write) periodically.  Drive fuse syncs each
        # write to the cloud, so every metrics_file.write+flush turns
        # into a network roundtrip.  Local /tmp file is ~microseconds.
        #
        # The local file is the source of truth during the run; we
        # copy to run_dir at:
        #   * end of every phase (after phase_summary written)
        #   * end of every rehearsal block
        #   * end of polish + consolidation phases
        # If the run crashes between flushes, the local file has
        # everything; user can `cp /tmp/<file> <run_dir>/` manually.
        metrics_path = run_dir / "metrics.jsonl"
        import tempfile
        _metrics_local_dir = Path(tempfile.gettempdir())
        _metrics_local_name = (
            f"nsos_metrics_{run_dir.name}_{int(time.time())}.jsonl"
        )
        metrics_local_path = _metrics_local_dir / _metrics_local_name
        def _flush_metrics_to_run_dir():
            """Atomically copy the local metrics file to run_dir."""
            if metrics_local_path.exists():
                shutil.copy2(metrics_local_path, metrics_path)
        supervised_history: List[Dict] = []
        supervised_history_by_family: Dict[str, List[Dict]] = {}
        replay_rng = random.Random(args.seed + 9001)
        base_learning_rate = float(profile["lr"])
        global_suite = build_global_suite(args.bundle_dir, args.global_suite_samples_per_phase)
        best_global_score = float("-inf")
        best_release_score = float("-inf")
        best_capacity_score = float("-inf")
        best_capacity_kind_scores: Dict[str, float] = {}
        # LEARN A2: resolve curriculum phase order from profile.  v11
        # profiles set curriculum_phase_order="v11" which orders as
        # curated_text→algorithms→structured→instructions→verifier→memory.
        execution_phase_order = resolve_phase_order(
            profile.get("curriculum_phase_order"))
        # AUDIT #8: open the LOCAL metrics file for writing.  Periodic
        # _flush_metrics_to_run_dir() copies it to the (possibly slow)
        # run_dir at phase boundaries.
        print(
            f"[boot] entering phase loop: order={execution_phase_order} "
            f"(per-phase tokenization is the canonical 'first 2-3 min silent' window)",
            flush=True,
        )
        with metrics_local_path.open("w", encoding="utf-8") as metrics_file:
            for phase_index, phase_name in enumerate(execution_phase_order):
                print(
                    f"[boot] phase {phase_index + 1}/{len(execution_phase_order)} = {phase_name}: "
                    f"loading rows from bundle...",
                    flush=True,
                )
                train_rows = curriculum_texts_for_phase(args.bundle_dir, phase_name, "train")
                eval_rows = curriculum_texts_for_phase(args.bundle_dir, phase_name, "eval")
                print(
                    f"[boot] phase {phase_name}: tokenizing {len(train_rows)} train rows "
                    f"(BPE on Python ~1-3 min for 30K+ rows)...",
                    flush=True,
                )
                train_tokens = build_token_stream(tokenizer, train_rows, "<|endoftext|>")
                print(
                    f"[boot] phase {phase_name}: tokenized to {len(train_tokens):,} tokens; eval_rows={len(eval_rows)}",
                    flush=True,
                )
                max_steps = args.override_phase_steps or profile["phase_steps"][phase_name]
                phase_lr_scale = float(profile.get("phase_lr_scale", {}).get(phase_name, 1.0))
                phase_repeat_scale = phase_repetition_scale(profile, phase_name)
                phase_aux_cfg = resolve_auxiliary_stack_config(profile, phase_name)
                apply_auxiliary_stack_schedule(trainer, phase_aux_cfg)
                trainer.phase_scheduler.auxiliary_memory_scope = (
                    ((int(args.seed) & 0xFFFF) << 8) + phase_index + 1
                )
                trainer.learning_rate = base_learning_rate * phase_lr_scale
                trainer.repetition_unlikelihood_scale = phase_repeat_scale
                phase_rows = list(train_rows)
                weak_kind_minipack_rows = build_weak_kind_minipack(
                    profile,
                    phase_name,
                    args.seed,
                )
                phase_rows.extend(weak_kind_minipack_rows)
                if phase_name != "phase3_curated_text":
                    phase_rows.extend(
                        sample_phase_replay_rows(
                            phase_name,
                            supervised_history_by_family,
                            args.replay_ratio,
                            len(train_rows),
                            replay_rng,
                            profile.get("phase_replay_config"),
                        )
                    )

                logger.log(
                    f"[train] {phase_name}: samples={len(train_rows)} mixed={len(phase_rows)} "
                    f"weak_pack={len(weak_kind_minipack_rows)} "
                    f"tokens={len(train_tokens)} steps={max_steps} seq_len={profile['seq_len']} "
                    f"batch={profile['batch_size']} lr={trainer.learning_rate:.2e} "
                    f"rul={trainer.repetition_unlikelihood_scale:.3f} "
                    f"aux={{enabled:{int(phase_aux_cfg['enabled'])},"
                    f"ttt:{int(phase_aux_cfg['session_adapt'])},"
                    f"reason:{int(phase_aux_cfg['reasoning'])},"
                    f"mem:{int(phase_aux_cfg['memory'])}}}"
                )
                set_layer_audit_phase(layer_audit, f"{phase_name}:train", int(trainer.global_step_count))
                phase_started = time.perf_counter()
                eval_subset = eval_rows[: max(1, args.phase_eval_samples)]
                ema_loss = None
                best_phase_score = float("-inf")
                best_phase_step = 0
                best_phase_name = ""
                best_phase_metrics: Dict[str, float] | None = None
                progress = logger.make_progress(max_steps, phase_name)

                def callback(step: int, loss: float) -> None:
                    nonlocal ema_loss, best_phase_score, best_phase_step, best_phase_name, best_phase_metrics
                    if ema_loss is None:
                        ema_loss = loss
                    else:
                        ema_loss = args.log_ema_beta * ema_loss + (1.0 - args.log_ema_beta) * loss

                    cur_lr = estimate_current_lr(trainer)
                    if progress is not None:
                        if step > progress.n:
                            progress.update(step - progress.n)
                        progress.set_postfix_str(
                            f"loss={loss:.4f} ema={ema_loss:.4f} lr={cur_lr:.2e}"
                        )
                    elif step == 1 or step % max(args.log_every_steps, 1) == 0 or step == max_steps:
                        logger.log(f"  step={step} loss={loss:.4f} ema={ema_loss:.4f} lr={cur_lr:.2e}")
                        if step == 1:
                            try:
                                _t = model.runtime_telemetry()
                                logger.log(
                                    f"[boot] mamba GPU fastpath: hits={_t['mamba_fast_path_hits']} "
                                    f"fallbacks={_t['mamba_fast_path_fallbacks']} "
                                    f"reason={_t['mamba_last_fallback_reason']!r}")
                            except Exception as _exc:  # binding antigo: reporta, nunca silencia
                                logger.log(f"[boot] runtime_telemetry indisponivel: {_exc}")

                    if args.checkpoint_every_steps > 0 and step % args.checkpoint_every_steps == 0:
                        save_checkpoint_artifact(f"{phase_name}_step{step}")
                    if (
                        args.phase_best_eval_every_steps > 0
                        and eval_subset
                        and (step % args.phase_best_eval_every_steps == 0 or step == max_steps)
                    ):
                        set_layer_audit_phase(layer_audit, f"{phase_name}:probe@{step}", step)
                        try:
                            probe_metrics = evaluate_phase(
                                nsos,
                                model,
                                tokenizer,
                                eval_subset,
                                eos_token_id,
                                profile["seq_len"],
                                args.phase_eval_mode,
                                args.phase_exact_samples,
                                logger=logger,
                                label=f"{phase_name}:probe@{step}",
                            )
                        finally:
                            set_layer_audit_phase(layer_audit, f"{phase_name}:train", step)
                        probe_score = compute_phase_score(probe_metrics)
                        logger.log(
                            f"  probe step={step} score={probe_score:.4f} "
                            f"answer_loss={probe_metrics['answer_loss']:.4f} "
                            f"first={probe_metrics['first_token_accuracy']:.2f} "
                            f"teacher={probe_metrics['teacher_token_accuracy']:.2f}"
                        )
                        if probe_score > best_phase_score:
                            best_phase_score = probe_score
                            best_phase_step = step
                            best_phase_metrics = dict(probe_metrics)
                            best_phase_name = f"{phase_name}_best_step{step}"
                            save_checkpoint_artifact(best_phase_name)
                            logger.log(f"  best checkpoint updated -> {best_phase_name}")

                try:
                    phase_aux_metrics: Dict[str, float] = {}
                    if phase_name == "phase3_curated_text":
                        trainer.train_loop(
                            train_tokens,
                            1,
                            profile["batch_size"],
                            profile["seq_len"],
                            callback,
                            max_steps,
                        )
                    else:
                        phase_aux_metrics = train_rows_direct(
                            trainer,
                            tokenizer,
                            phase_rows,
                            max_steps,
                            callback,
                            seed=args.seed + sum(ord(ch) for ch in phase_name),
                            eos_token_id=eos_token_id,
                            batch_size=profile["batch_size"],
                        )
                finally:
                    if progress is not None:
                        if progress.n < max_steps:
                            progress.update(max_steps - progress.n)
                        progress.close()

                if phase_name != "phase3_curated_text":
                    history_rows = list(train_rows) + list(weak_kind_minipack_rows)
                    supervised_history.extend(history_rows)
                    family = PHASE_FAMILIES.get(phase_name)
                    if family:
                        supervised_history_by_family.setdefault(family, []).extend(history_rows)

                selected_source = "final"
                if best_phase_name:
                    debug_eval_log(logger, f"{phase_name}:loading_best:start {best_phase_name}")
                    model.load(str(run_dir / f"{best_phase_name}.bin"))
                    debug_eval_log(logger, f"{phase_name}:loading_best:done {best_phase_name}")
                    save_checkpoint_artifact(f"{phase_name}_best")
                    selected_source = "best_eval"
                    set_layer_audit_phase(layer_audit, f"{phase_name}:eval", best_phase_step or max_steps)
                    phase_metrics = (
                        dict(best_phase_metrics)
                        if best_phase_metrics is not None and layer_audit is None
                        else evaluate_phase(
                            nsos,
                            model,
                            tokenizer,
                            eval_subset,
                            eos_token_id,
                            profile["seq_len"],
                            args.phase_eval_mode,
                            args.phase_exact_samples,
                            logger=logger,
                            label=f"{phase_name}:best_eval_reload",
                        )
                    )
                else:
                    set_layer_audit_phase(layer_audit, f"{phase_name}:eval", max_steps)
                    phase_metrics = evaluate_phase(
                        nsos,
                        model,
                        tokenizer,
                        eval_subset,
                        eos_token_id,
                        profile["seq_len"],
                        args.phase_eval_mode,
                        args.phase_exact_samples,
                        logger=logger,
                        label=f"{phase_name}:final_eval",
                    )
                phase_elapsed = time.perf_counter() - phase_started

                phase_summary = {
                    "phase": phase_name,
                    "train_samples": len(train_rows),
                    "mixed_samples": len(phase_rows),
                    "weak_kind_minipack_samples": len(weak_kind_minipack_rows),
                    "eval_samples": len(eval_subset),
                    "train_tokens": len(train_tokens),
                    "max_steps": max_steps,
                    "elapsed_s": phase_elapsed,
                    "ema_loss_final": ema_loss if ema_loss is not None else 0.0,
                    "repetition_unlikelihood_scale": phase_repeat_scale,
                    "auxiliary_stack": phase_aux_cfg,
                    "auxiliary_stack_metrics": phase_aux_metrics,
                    "selected_checkpoint_source": selected_source,
                    "best_step": best_phase_step,
                    "best_score": best_phase_score if best_phase_score != float("-inf") else 0.0,
                    **phase_metrics,
                }
                should_run_global_suite = (
                    args.global_suite_every_phases > 0
                    and (
                        ((phase_index + 1) % args.global_suite_every_phases == 0)
                        or phase_index == len(PHASE_ORDER) - 1
                    )
                )
                if should_run_global_suite:
                    phase_summary["global_suite_score"] = consider_global_champion(
                        f"{phase_name}:{selected_source}@{phase_summary['best_step'] or max_steps}",
                        phase_name,
                    )
                else:
                    phase_summary["global_suite_score"] = None
                    logger.log(
                        f"[global] skipped after {phase_name}; cadence={args.global_suite_every_phases}"
                    )
                phase_summary["global_suite_evaluated"] = should_run_global_suite
                should_run_capacity_holdout = (
                    args.holdout_every_phases > 0
                    and (
                        ((phase_index + 1) % args.holdout_every_phases == 0)
                        or phase_index == len(PHASE_ORDER) - 1
                    )
                )
                if should_run_capacity_holdout:
                    phase_summary["capacity_holdout_score"] = consider_capacity_champion(
                        f"{phase_name}:{selected_source}@{phase_summary['best_step'] or max_steps}",
                        phase_name,
                    )
                else:
                    phase_summary["capacity_holdout_score"] = None
                phase_summary["capacity_holdout_evaluated"] = should_run_capacity_holdout
                if layer_audit is not None:
                    phase_summary["layer_audit"] = {
                        "train": layer_audit_summary_to_dict(
                            layer_audit.summarize_phase(f"{phase_name}:train")
                        ),
                        "eval": layer_audit_summary_to_dict(
                            layer_audit.summarize_phase(f"{phase_name}:eval")
                        ),
                        "global": layer_audit_summary_to_dict(
                            layer_audit.summarize_phase(f"{phase_name}:global")
                        ),
                    }
                    write_layer_audit_snapshot(layer_audit, layer_audit_path)
                    summary["layer_audit"]["phase_snapshots"].append(
                        {
                            "phase": phase_name,
                            "path": str(layer_audit_path),
                            "train_records": phase_summary["layer_audit"]["train"].get("records", 0),
                            "eval_records": phase_summary["layer_audit"]["eval"].get("records", 0),
                        }
                    )
                summary["phases"].append(phase_summary)
                metrics_file.write(json.dumps(phase_summary, ensure_ascii=False) + "\n")
                metrics_file.flush()
                # AUDIT #8: copy local metrics to run_dir at phase boundary.
                _flush_metrics_to_run_dir()

                if selected_source == "final":
                    save_checkpoint_artifact(phase_name)
                logger.log(
                    f"[eval] {phase_name}: answer_loss={phase_summary['answer_loss']:.4f} "
                    f"first={phase_summary['first_token_accuracy']:.2f} "
                    f"teacher={phase_summary['teacher_token_accuracy']:.2f} "
                    f"exact={phase_summary['exact_correct']}/{phase_summary['exact_total']} "
                    f"selected={selected_source}@{phase_summary['best_step'] or max_steps}"
                )
                if phase_aux_cfg["enabled"]:
                    logger.log(f"[aux] {phase_name}: {format_auxiliary_metrics(phase_aux_metrics)}")

            weak_kind_rehearsal_steps = int(profile.get("weak_kind_rehearsal_steps", 0))
            if weak_kind_rehearsal_steps > 0 and args.override_phase_steps <= 0:
                rehearsal_phase_name = "weak_kind_rehearsal"
                rehearsal_rows = build_weak_kind_minipack(
                    profile,
                    rehearsal_phase_name,
                    args.seed,
                )
                if rehearsal_rows:
                    rehearsal_lr_scale = float(profile.get("weak_kind_rehearsal_lr_scale", 1.0))
                    trainer.learning_rate = base_learning_rate * rehearsal_lr_scale
                    trainer.repetition_unlikelihood_scale = 0.0
                    apply_auxiliary_stack_schedule(
                        trainer,
                        resolve_auxiliary_stack_config(profile, rehearsal_phase_name),
                    )
                    logger.log(
                        f"[train] {rehearsal_phase_name}: rows={len(rehearsal_rows)} "
                        f"steps={weak_kind_rehearsal_steps} batch={profile['batch_size']} "
                        f"lr={trainer.learning_rate:.2e}"
                    )
                    set_layer_audit_phase(
                        layer_audit,
                        f"{rehearsal_phase_name}:train",
                        int(trainer.global_step_count),
                    )
                    rehearsal_started = time.perf_counter()
                    rehearsal_ema = None
                    rehearsal_progress = logger.make_progress(
                        weak_kind_rehearsal_steps,
                        rehearsal_phase_name,
                    )

                    def rehearsal_callback(step: int, loss: float) -> None:
                        nonlocal rehearsal_ema
                        if rehearsal_ema is None:
                            rehearsal_ema = loss
                        else:
                            rehearsal_ema = (
                                args.log_ema_beta * rehearsal_ema
                                + (1.0 - args.log_ema_beta) * loss
                            )
                        cur_lr = estimate_current_lr(trainer)
                        if rehearsal_progress is not None:
                            if step > rehearsal_progress.n:
                                rehearsal_progress.update(step - rehearsal_progress.n)
                            rehearsal_progress.set_postfix_str(
                                f"loss={loss:.4f} ema={rehearsal_ema:.4f} lr={cur_lr:.2e}"
                            )
                        elif (
                            step == 1
                            or step % max(args.log_every_steps, 1) == 0
                            or step == weak_kind_rehearsal_steps
                        ):
                            logger.log(
                                f"  step={step} loss={loss:.4f} "
                                f"ema={rehearsal_ema:.4f} lr={cur_lr:.2e}"
                            )

                    try:
                        rehearsal_aux_metrics = train_rows_direct(
                            trainer,
                            tokenizer,
                            rehearsal_rows,
                            weak_kind_rehearsal_steps,
                            rehearsal_callback,
                            seed=args.seed + 919191,
                            eos_token_id=eos_token_id,
                            batch_size=profile["batch_size"],
                        )
                    finally:
                        if rehearsal_progress is not None:
                            if rehearsal_progress.n < weak_kind_rehearsal_steps:
                                rehearsal_progress.update(
                                    weak_kind_rehearsal_steps - rehearsal_progress.n
                                )
                            rehearsal_progress.close()

                    rehearsal_eval_rows = rehearsal_rows[: max(1, args.phase_eval_samples * 2)]
                    set_layer_audit_phase(
                        layer_audit,
                        f"{rehearsal_phase_name}:eval",
                        weak_kind_rehearsal_steps,
                    )
                    rehearsal_metrics = evaluate_phase(
                        nsos,
                        model,
                        tokenizer,
                        rehearsal_eval_rows,
                        eos_token_id,
                        profile["seq_len"],
                        args.phase_eval_mode,
                        args.phase_exact_samples,
                        logger=logger,
                        label=f"{rehearsal_phase_name}:final_eval",
                    )
                    rehearsal_summary = {
                        "phase": rehearsal_phase_name,
                        "train_samples": len(rehearsal_rows),
                        "mixed_samples": len(rehearsal_rows),
                        "weak_kind_minipack_samples": len(rehearsal_rows),
                        "eval_samples": len(rehearsal_eval_rows),
                        "train_tokens": 0,
                        "max_steps": weak_kind_rehearsal_steps,
                        "elapsed_s": time.perf_counter() - rehearsal_started,
                        "ema_loss_final": rehearsal_ema if rehearsal_ema is not None else 0.0,
                        "auxiliary_stack_metrics": rehearsal_aux_metrics,
                        **rehearsal_metrics,
                    }
                    rehearsal_summary["global_suite_score"] = None
                    rehearsal_summary["global_suite_evaluated"] = False
                    if args.holdout_every_phases > 0:
                        rehearsal_summary["capacity_holdout_score"] = consider_capacity_champion(
                            f"{rehearsal_phase_name}:final@{weak_kind_rehearsal_steps}",
                            rehearsal_phase_name,
                        )
                        rehearsal_summary["capacity_holdout_evaluated"] = True
                    else:
                        rehearsal_summary["capacity_holdout_score"] = None
                        rehearsal_summary["capacity_holdout_evaluated"] = False
                    if layer_audit is not None:
                        rehearsal_summary["layer_audit"] = {
                            "train": layer_audit_summary_to_dict(
                                layer_audit.summarize_phase(f"{rehearsal_phase_name}:train")
                            ),
                            "eval": layer_audit_summary_to_dict(
                                layer_audit.summarize_phase(f"{rehearsal_phase_name}:eval")
                            ),
                            "global": {},
                        }
                        write_layer_audit_snapshot(layer_audit, layer_audit_path)
                        summary["layer_audit"]["phase_snapshots"].append(
                            {
                                "phase": rehearsal_phase_name,
                                "path": str(layer_audit_path),
                                "train_records": rehearsal_summary["layer_audit"]["train"].get("records", 0),
                                "eval_records": rehearsal_summary["layer_audit"]["eval"].get("records", 0),
                            }
                        )
                    summary["phases"].append(rehearsal_summary)
                    metrics_file.write(json.dumps(rehearsal_summary, ensure_ascii=False) + "\n")
                    metrics_file.flush()
                    _flush_metrics_to_run_dir()
                    save_checkpoint_artifact(rehearsal_phase_name)
                    trainer.learning_rate = base_learning_rate
                    logger.log(
                        f"[eval] {rehearsal_phase_name}: "
                        f"answer_loss={rehearsal_summary['answer_loss']:.4f} "
                        f"first={rehearsal_summary['first_token_accuracy']:.2f} "
                        f"teacher={rehearsal_summary['teacher_token_accuracy']:.2f}"
                    )

            instruction_polish_steps = int(profile.get("instruction_polish_steps", 0))
            if instruction_polish_steps > 0 and args.override_phase_steps <= 0:
                polish_phase_name = "instruction_polish"
                polish_rows = (
                    curriculum_texts_for_phase(args.bundle_dir, "phase4_instructions", "train")
                    + curriculum_texts_for_phase(args.bundle_dir, "phase5_verifier", "train")
                    + curriculum_texts_for_phase(args.bundle_dir, "phase6_memory", "train")
                )
                polish_eval_rows = (
                    curriculum_texts_for_phase(args.bundle_dir, "phase4_instructions", "eval")
                    + curriculum_texts_for_phase(args.bundle_dir, "phase5_verifier", "eval")
                    + curriculum_texts_for_phase(args.bundle_dir, "phase6_memory", "eval")
                )[: max(1, args.phase_eval_samples * 2)]
                polish_lr_scale = float(profile.get("instruction_polish_lr_scale", 1.0))
                polish_lr = base_learning_rate * polish_lr_scale
                polish_repeat_scale = phase_repetition_scale(profile, polish_phase_name)
                polish_aux_cfg = resolve_auxiliary_stack_config(profile, polish_phase_name)
                apply_auxiliary_stack_schedule(trainer, polish_aux_cfg)
                trainer.phase_scheduler.auxiliary_memory_scope = (
                    ((int(args.seed) & 0xFFFF) << 8) + len(PHASE_ORDER) + 1
                )
                trainer.learning_rate = polish_lr
                trainer.repetition_unlikelihood_scale = polish_repeat_scale

                logger.log(
                    f"[train] {polish_phase_name}: rows={len(polish_rows)} "
                    f"steps={instruction_polish_steps} batch={profile['batch_size']} "
                    f"lr={trainer.learning_rate:.2e} "
                    f"rul={trainer.repetition_unlikelihood_scale:.3f} "
                    f"aux={{enabled:{int(polish_aux_cfg['enabled'])},"
                    f"ttt:{int(polish_aux_cfg['session_adapt'])},"
                    f"reason:{int(polish_aux_cfg['reasoning'])},"
                    f"mem:{int(polish_aux_cfg['memory'])}}}"
                )
                set_layer_audit_phase(layer_audit, f"{polish_phase_name}:train", int(trainer.global_step_count))
                polish_started = time.perf_counter()
                polish_ema = None
                polish_best_score = float("-inf")
                polish_best_step = 0
                polish_best_name = ""
                polish_best_metrics: Dict[str, float] | None = None
                polish_aux_metrics: Dict[str, float] = {}
                polish_progress = logger.make_progress(instruction_polish_steps, polish_phase_name)

                def polish_callback(step: int, loss: float) -> None:
                    nonlocal polish_ema, polish_best_score, polish_best_step, polish_best_name
                    nonlocal polish_best_metrics
                    if polish_ema is None:
                        polish_ema = loss
                    else:
                        polish_ema = args.log_ema_beta * polish_ema + (1.0 - args.log_ema_beta) * loss

                    cur_lr = estimate_current_lr(trainer)
                    if polish_progress is not None:
                        if step > polish_progress.n:
                            polish_progress.update(step - polish_progress.n)
                        polish_progress.set_postfix_str(
                            f"loss={loss:.4f} ema={polish_ema:.4f} lr={cur_lr:.2e}"
                        )
                    elif (
                        step == 1
                        or step % max(args.log_every_steps, 1) == 0
                        or step == instruction_polish_steps
                    ):
                        logger.log(
                            f"  step={step} loss={loss:.4f} ema={polish_ema:.4f} lr={cur_lr:.2e}"
                        )

                    if (
                        args.phase_best_eval_every_steps > 0
                        and polish_eval_rows
                        and (
                            step % args.phase_best_eval_every_steps == 0
                            or step == instruction_polish_steps
                        )
                    ):
                        set_layer_audit_phase(layer_audit, f"{polish_phase_name}:probe@{step}", step)
                        try:
                            probe_metrics = evaluate_phase(
                                nsos,
                                model,
                                tokenizer,
                                polish_eval_rows,
                                eos_token_id,
                                profile["seq_len"],
                                args.phase_eval_mode,
                                args.phase_exact_samples,
                                logger=logger,
                                label=f"{polish_phase_name}:probe@{step}",
                            )
                        finally:
                            set_layer_audit_phase(layer_audit, f"{polish_phase_name}:train", step)
                        probe_score = compute_phase_score(probe_metrics)
                        logger.log(
                            f"  probe step={step} score={probe_score:.4f} "
                            f"answer_loss={probe_metrics['answer_loss']:.4f} "
                            f"first={probe_metrics['first_token_accuracy']:.2f} "
                            f"teacher={probe_metrics['teacher_token_accuracy']:.2f}"
                        )
                        if probe_score > polish_best_score:
                            polish_best_score = probe_score
                            polish_best_step = step
                            polish_best_metrics = dict(probe_metrics)
                            polish_best_name = f"{polish_phase_name}_best_step{step}"
                            save_checkpoint_artifact(polish_best_name)
                            logger.log(f"  best checkpoint updated -> {polish_best_name}")

                try:
                    polish_aux_metrics = train_rows_direct(
                        trainer,
                        tokenizer,
                        polish_rows,
                        instruction_polish_steps,
                        polish_callback,
                        seed=args.seed + 818181,
                        eos_token_id=eos_token_id,
                        batch_size=profile["batch_size"],
                    )
                finally:
                    if polish_progress is not None:
                        if polish_progress.n < instruction_polish_steps:
                            polish_progress.update(instruction_polish_steps - polish_progress.n)
                        polish_progress.close()

                selected_source = "final"
                if polish_best_name:
                    debug_eval_log(logger, f"{polish_phase_name}:loading_best:start {polish_best_name}")
                    model.load(str(run_dir / f"{polish_best_name}.bin"))
                    debug_eval_log(logger, f"{polish_phase_name}:loading_best:done {polish_best_name}")
                    save_checkpoint_artifact(f"{polish_phase_name}_best")
                    selected_source = "best_eval"
                    set_layer_audit_phase(layer_audit, f"{polish_phase_name}:eval", polish_best_step or instruction_polish_steps)
                    polish_metrics = dict(polish_best_metrics) if polish_best_metrics is not None and layer_audit is None else evaluate_phase(
                        nsos,
                        model,
                        tokenizer,
                        polish_eval_rows,
                        eos_token_id,
                        profile["seq_len"],
                        args.phase_eval_mode,
                        args.phase_exact_samples,
                        logger=logger,
                        label=f"{polish_phase_name}:best_eval_reload",
                    )
                else:
                    set_layer_audit_phase(layer_audit, f"{polish_phase_name}:eval", instruction_polish_steps)
                    polish_metrics = evaluate_phase(
                        nsos,
                        model,
                        tokenizer,
                        polish_eval_rows,
                        eos_token_id,
                        profile["seq_len"],
                        args.phase_eval_mode,
                        args.phase_exact_samples,
                        logger=logger,
                        label=f"{polish_phase_name}:final_eval",
                    )

                polish_summary = {
                    "phase": polish_phase_name,
                    "train_samples": len(polish_rows),
                    "mixed_samples": len(polish_rows),
                    "eval_samples": len(polish_eval_rows),
                    "train_tokens": len(build_token_stream(tokenizer, polish_rows, "<|endoftext|>")),
                    "max_steps": instruction_polish_steps,
                    "elapsed_s": time.perf_counter() - polish_started,
                    "ema_loss_final": polish_ema if polish_ema is not None else 0.0,
                    "repetition_unlikelihood_scale": polish_repeat_scale,
                    "auxiliary_stack": polish_aux_cfg,
                    "auxiliary_stack_metrics": polish_aux_metrics,
                    "selected_checkpoint_source": selected_source,
                    "best_step": polish_best_step,
                    "best_score": polish_best_score if polish_best_score != float("-inf") else 0.0,
                    **polish_metrics,
                }
                polish_summary["global_suite_score"] = consider_global_champion(
                    f"{polish_phase_name}:{selected_source}@{polish_summary['best_step'] or instruction_polish_steps}",
                    polish_phase_name,
                )
                polish_summary["global_suite_evaluated"] = True
                if args.holdout_every_phases > 0:
                    polish_summary["capacity_holdout_score"] = consider_capacity_champion(
                        f"{polish_phase_name}:{selected_source}@{polish_summary['best_step'] or instruction_polish_steps}",
                        polish_phase_name,
                    )
                    polish_summary["capacity_holdout_evaluated"] = True
                else:
                    polish_summary["capacity_holdout_score"] = None
                    polish_summary["capacity_holdout_evaluated"] = False
                if layer_audit is not None:
                    polish_summary["layer_audit"] = {
                        "train": layer_audit_summary_to_dict(
                            layer_audit.summarize_phase(f"{polish_phase_name}:train")
                        ),
                        "eval": layer_audit_summary_to_dict(
                            layer_audit.summarize_phase(f"{polish_phase_name}:eval")
                        ),
                        "global": layer_audit_summary_to_dict(
                            layer_audit.summarize_phase(f"{polish_phase_name}:global")
                        ),
                    }
                    write_layer_audit_snapshot(layer_audit, layer_audit_path)
                    summary["layer_audit"]["phase_snapshots"].append(
                        {
                            "phase": polish_phase_name,
                            "path": str(layer_audit_path),
                            "train_records": polish_summary["layer_audit"]["train"].get("records", 0),
                            "eval_records": polish_summary["layer_audit"]["eval"].get("records", 0),
                        }
                    )
                summary["phases"].append(polish_summary)
                metrics_file.write(json.dumps(polish_summary, ensure_ascii=False) + "\n")
                metrics_file.flush()
                _flush_metrics_to_run_dir()
                if selected_source == "final":
                    save_checkpoint_artifact(polish_phase_name)
                logger.log(
                    f"[eval] {polish_phase_name}: answer_loss={polish_summary['answer_loss']:.4f} "
                    f"first={polish_summary['first_token_accuracy']:.2f} "
                    f"teacher={polish_summary['teacher_token_accuracy']:.2f} "
                    f"exact={polish_summary['exact_correct']}/{polish_summary['exact_total']} "
                    f"selected={selected_source}@{polish_summary['best_step'] or instruction_polish_steps}"
                )
                if polish_aux_cfg["enabled"]:
                    logger.log(
                        f"[aux] {polish_phase_name}: {format_auxiliary_metrics(polish_aux_metrics)}"
                    )
                trainer.learning_rate = base_learning_rate

            if args.final_consolidation_steps > 0 and supervised_history:
                logger.log(
                    f"[train] consolidation: rows={len(supervised_history)} "
                    f"steps={args.final_consolidation_steps} batch={profile['batch_size']}"
                )
                set_layer_audit_phase(layer_audit, "final_consolidation:train", int(trainer.global_step_count))
                consolidation_started = time.perf_counter()
                consolidation_ema = None
                consolidation_aux_metrics: Dict[str, float] = {}
                consolidation_progress = logger.make_progress(
                    args.final_consolidation_steps,
                    "consolidation",
                )

                def final_callback(step: int, loss: float) -> None:
                    nonlocal consolidation_ema
                    if consolidation_ema is None:
                        consolidation_ema = loss
                    else:
                        consolidation_ema = (
                            args.log_ema_beta * consolidation_ema + (1.0 - args.log_ema_beta) * loss
                        )
                    cur_lr = estimate_current_lr(trainer)
                    if consolidation_progress is not None:
                        if step > consolidation_progress.n:
                            consolidation_progress.update(step - consolidation_progress.n)
                        consolidation_progress.set_postfix_str(
                            f"loss={loss:.4f} ema={consolidation_ema:.4f} lr={cur_lr:.2e}"
                        )
                    elif (
                        step == 1
                        or step % max(args.log_every_steps, 1) == 0
                        or step == args.final_consolidation_steps
                    ):
                        logger.log(
                            f"  step={step} loss={loss:.4f} ema={consolidation_ema:.4f} lr={cur_lr:.2e}"
                        )
                    if args.checkpoint_every_steps > 0 and step % args.checkpoint_every_steps == 0:
                        save_checkpoint_artifact(f"consolidation_step{step}")

                try:
                    consolidation_aux_metrics = train_rows_direct(
                        trainer,
                        tokenizer,
                        supervised_history,
                        args.final_consolidation_steps,
                        final_callback,
                        seed=args.seed + 424242,
                        eos_token_id=eos_token_id,
                        batch_size=profile["batch_size"],
                    )
                finally:
                    if consolidation_progress is not None:
                        if consolidation_progress.n < args.final_consolidation_steps:
                            consolidation_progress.update(
                                args.final_consolidation_steps - consolidation_progress.n
                            )
                        consolidation_progress.close()

                consolidation_eval_rows: List[Dict] = []
                for phase_name in resolve_phase_order(None):
                    if phase_name == "phase3_curated_text":
                        continue
                    consolidation_eval_rows.extend(
                        curriculum_texts_for_phase(args.bundle_dir, phase_name, "eval")[
                            : max(1, args.phase_eval_samples // 2)
                        ]
                    )
                set_layer_audit_phase(layer_audit, "final_consolidation:eval", args.final_consolidation_steps)
                consolidation_metrics = evaluate_phase(
                    nsos,
                    model,
                    tokenizer,
                    consolidation_eval_rows,
                    eos_token_id,
                    profile["seq_len"],
                    args.phase_eval_mode,
                    args.phase_exact_samples,
                    logger=logger,
                    label="final_consolidation:final_eval",
                )
                consolidation_summary = {
                    "phase": "final_consolidation",
                    "train_samples": len(supervised_history),
                    "mixed_samples": len(supervised_history),
                    "eval_samples": len(consolidation_eval_rows),
                    "train_tokens": 0,
                    "max_steps": args.final_consolidation_steps,
                    "elapsed_s": time.perf_counter() - consolidation_started,
                    "ema_loss_final": consolidation_ema if consolidation_ema is not None else 0.0,
                    "auxiliary_stack_metrics": consolidation_aux_metrics,
                    **consolidation_metrics,
                }
                consolidation_summary["global_suite_score"] = consider_global_champion(
                    "final_consolidation",
                    "final_consolidation",
                )
                consolidation_summary["global_suite_evaluated"] = True
                if args.holdout_every_phases > 0:
                    consolidation_summary["capacity_holdout_score"] = consider_capacity_champion(
                        "final_consolidation",
                        "final_consolidation",
                    )
                    consolidation_summary["capacity_holdout_evaluated"] = True
                else:
                    consolidation_summary["capacity_holdout_score"] = None
                    consolidation_summary["capacity_holdout_evaluated"] = False
                if layer_audit is not None:
                    consolidation_summary["layer_audit"] = {
                        "train": layer_audit_summary_to_dict(
                            layer_audit.summarize_phase("final_consolidation:train")
                        ),
                        "eval": layer_audit_summary_to_dict(
                            layer_audit.summarize_phase("final_consolidation:eval")
                        ),
                        "global": layer_audit_summary_to_dict(
                            layer_audit.summarize_phase("final_consolidation:global")
                        ),
                    }
                    write_layer_audit_snapshot(layer_audit, layer_audit_path)
                    summary["layer_audit"]["phase_snapshots"].append(
                        {
                            "phase": "final_consolidation",
                            "path": str(layer_audit_path),
                            "train_records": consolidation_summary["layer_audit"]["train"].get("records", 0),
                            "eval_records": consolidation_summary["layer_audit"]["eval"].get("records", 0),
                        }
                    )
                summary["phases"].append(consolidation_summary)
                metrics_file.write(json.dumps(consolidation_summary, ensure_ascii=False) + "\n")
                metrics_file.flush()
                _flush_metrics_to_run_dir()
                save_checkpoint_artifact("final_consolidation")
                logger.log(
                    f"[eval] final_consolidation: answer_loss={consolidation_summary['answer_loss']:.4f} "
                    f"first={consolidation_summary['first_token_accuracy']:.2f} "
                    f"teacher={consolidation_summary['teacher_token_accuracy']:.2f}"
                )

        final_model_source = "current"
        release_candidate_path = run_dir / "release_candidate.bin"
        holdout_champion_path = run_dir / "champion_holdout.bin"
        champion_path = run_dir / "champion_global.bin"
        if summary.get("release_candidate", {}).get("source") and release_candidate_path.exists():
            model.load(str(release_candidate_path))
            final_model_source = "release_candidate"
        elif (
            args.select_final_by_holdout
            and summary.get("capacity_champion", {}).get("source")
            and holdout_champion_path.exists()
        ):
            model.load(str(holdout_champion_path))
            final_model_source = "champion_holdout"
        elif summary.get("global_champion", {}).get("source") and champion_path.exists():
            model.load(str(champion_path))
            final_model_source = "champion_global"
        summary["final_model_source"] = final_model_source

        summary["official_holdouts"] = evaluate_official_holdouts(
            nsos,
            model,
            tokenizer,
            list(args.holdout_files or []),
            eos_token_id,
            profile["seq_len"],
            args.holdout_eval_mode,
            args.holdout_exact_samples,
            logger,
            layer_audit,
        )
        summary["official_holdout_kind_breakdown"] = aggregate_holdout_kind_breakdown(
            summary["official_holdouts"]
        )
        summary["capacity_best_kind_scores"] = dict(best_capacity_kind_scores)
        summary["capacity_kind_regression_gate"] = evaluate_kind_regression_gate(
            summary["official_holdout_kind_breakdown"],
            best_capacity_kind_scores,
            profile.get("kind_regression_gate", {}),
        )
        capacity_gate_failed = not bool(summary["capacity_kind_regression_gate"].get("passed", True))
        if summary["capacity_kind_regression_gate"].get("enabled", False):
            logger.log(
                f"[capacity-gate] passed={int(not capacity_gate_failed)} "
                f"failures={summary['capacity_kind_regression_gate'].get('failure_count', 0)}"
            )

        audit_gate_failed = False
        if layer_audit is not None:
            reload_probe_rows: List[Dict] = []
            for holdout_path in list(args.holdout_files or []):
                reload_probe_rows.extend(load_holdout_jsonl(holdout_path))
            reload_probe = run_reload_probe(
                nsos,
                model,
                model_config,
                device,
                tokenizer,
                reload_probe_rows,
                eos_token_id,
                run_dir,
                layer_audit,
                logger,
                sample_limit=max(1, min(args.holdout_exact_samples, 4)),
            )
            summary["layer_audit"]["reload_probe"] = reload_probe
            write_layer_audit_snapshot(layer_audit, layer_audit_path)
            inline_run_summary = {
                "path": str(run_dir / "run_summary.json"),
                "profile": canonical_profile,
                "requested_profile": args.profile,
                "device": summary["device"],
                "official_holdouts": summary["official_holdouts"],
                "reload_probe": reload_probe,
            }
            audit_report = build_layer_audit_report(
                [layer_audit_path],
                metrics_paths=[metrics_path],
                thresholds=audit_thresholds,
                run_summaries=[inline_run_summary],
            )
            layer_audit_summary_path.parent.mkdir(parents=True, exist_ok=True)
            layer_audit_summary_path.write_text(
                json.dumps(audit_report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            summary["layer_audit"]["gate"] = {
                "passed": audit_report["verdict"] == "pass",
                "failure_count": len(audit_report.get("failures", [])),
                "failures": audit_report.get("failures", []),
            }
            summary["layer_audit"]["totals"] = (
                audit_report.get("audits", [{}])[0].get("totals", {})
            )
            summary["layer_audit"]["report_path"] = str(layer_audit_summary_path)
            audit_gate_failed = audit_report["verdict"] != "pass"
            logger.log(
                f"[audit] gate={audit_report['verdict']} "
                f"failures={len(audit_report.get('failures', []))} "
                f"summary={layer_audit_summary_path}"
            )
        model.save(str(run_dir / "final_model.bin"))
        model.save_edge_linear_pack(str(run_dir / "final_edge_linear.nsos"))
        save_run_summary(run_dir / "run_summary.json", summary)
        logger.log(f"[done] final checkpoint: {run_dir / 'final_model.bin'}")
        logger.log(f"[done] final edge pack: {run_dir / 'final_edge_linear.nsos'}")
        logger.log(f"[done] summary: {run_dir / 'run_summary.json'}")
        if audit_gate_failed:
            return 2
        if capacity_gate_failed:
            return 3
        return 0
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
