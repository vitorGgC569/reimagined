from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, TextIO

from cuda_env import add_windows_runtime_dirs, parse_preferred_cuda_root
from nsos_curriculum_lib import (
    PHASE_ORDER,
    SPECIAL_TOKENS,
    build_curriculum,
    build_tokenizer_bundle,
    curriculum_texts_for_phase,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


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
            "enabled": False,
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
            "enabled": False,
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
        default=16,
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
        default=8,
        help="Number of held-out samples used for greedy exact-match when eval mode is full.",
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
        default=1,
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
        default=8,
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
    return parser.parse_args()


def detect_build_dir(explicit: Path | None) -> Path:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.extend([explicit, explicit / "Release"])
    repo_root = Path(__file__).resolve().parents[3]
    nsos_root = repo_root / "OXN" / "nsos"
    for name in ["build_cuda129", "build_v1", "build_full", "build_codex", "build"]:
        candidates.extend([nsos_root / name / "Release", nsos_root / name])

    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("nsos_ext*.pyd")):
            return candidate
    raise RuntimeError("Could not find a build directory with nsos_ext.")


def load_nsos(build_dir: Path):
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
        logits = None
        for prompt_index, token in enumerate(token_ids):
            logits = model.forward_ids([token], None)
            if prompt_index == 0 or (prompt_index + 1) == len(token_ids):
                debug_eval_log(logger, f"{label}:prompt_step={prompt_index + 1}/{len(token_ids)}")
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


def evaluate_masked_supervised(nsos, model, tokenizer, rows: List[Dict], eos_token_id: int) -> Dict[str, float]:
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

    for row in rows:
        if not row.get("answer"):
            continue
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
        loss, _ = host_logits.cross_entropy(answer_tokens)
        total_loss += float(loss)

        values = host_logits.numpy()
        if len(answer_tokens) > 0:
            if int(values[0].argmax()) == int(answer_tokens[0]):
                first_hits += 1

        for row_index, token in enumerate(answer_tokens):
            teacher_total += 1
            if int(values[row_index].argmax()) == int(token):
                teacher_hits += 1

    sample_count = max(sum(1 for row in rows if row.get("answer")), 1)
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
) -> List[Dict]:
    config = PHASE_REPLAY_CONFIG.get(phase_name, {"families": [], "ratio_scale": 0.0})
    effective_ratio = float(ratio) * float(config.get("ratio_scale", 0.0))
    families = list(config.get("families", []))
    if effective_ratio <= 0.0 or current_count <= 0 or not families:
        return []

    pool: List[Dict] = []
    for family in families:
        pool.extend(history_by_family.get(family, []))
    if not pool:
        return []

    replay_count = min(len(pool), max(1, int(current_count * effective_ratio)))
    if replay_count >= len(pool):
        picked = list(pool)
        rng.shuffle(picked)
        return picked
    return rng.sample(pool, replay_count)


def evaluate_generation_probe(nsos, model, tokenizer, rows: List[Dict], eos_token_id: int,
                              logger=None, label: str = "probe") -> Dict[str, float | str]:
    probe_rows = [row for row in rows if row.get("answer")][: max(1, min(4, len(rows)))]
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


def evaluate_phase(nsos, model, tokenizer, rows: List[Dict], eos_token_id: int,
                   seq_len: int, eval_mode: str, exact_samples: int,
                   logger=None, label: str = "phase_eval") -> Dict[str, float]:
    debug_eval_log(logger, f"{label}:masked:start rows={len(rows)}")
    masked_metrics = evaluate_masked_supervised(nsos, model, tokenizer, rows, eos_token_id)
    debug_eval_log(logger, f"{label}:masked:done")
    debug_eval_log(logger, f"{label}:text:start")
    text_metrics = evaluate_text_loss(nsos, model, tokenizer, rows, seq_len)
    debug_eval_log(logger, f"{label}:text:done")
    result = {**masked_metrics, **text_metrics}

    effective_exact_samples = max(1, exact_samples if eval_mode == "full" else min(exact_samples, 2))
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
        )
    )
    debug_eval_log(logger, f"{label}:probe:done")

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


def build_global_suite(bundle_dir: Path, samples_per_phase: int) -> Dict[str, List[Dict]]:
    suite: Dict[str, List[Dict]] = {}
    for phase_name in PHASE_ORDER:
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


def main() -> int:
    args = parse_args()
    canonical_profile, profile = resolve_profile(args.profile)
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    session_log_path = args.session_log or (run_dir / "session.log")
    logger = RunLogger(args.progress_mode, session_log_path)

    try:
        maybe_enable_hybrid_resume_cuda_safe_mode(args, profile, logger)
        build_dir = detect_build_dir(args.build_dir)
        nsos = load_nsos(build_dir)

        tokenizer_path = ensure_bundle(
            args.repo_root,
            args.bundle_dir,
            seed=args.seed,
            target_vocab=profile["target_vocab"],
            rebuild=args.rebuild_curriculum,
            phase_sizes=profile.get("phase_sizes"),
        )

        tokenizer = nsos.Tokenizer()
        tokenizer.load(str(tokenizer_path))
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
        tokenizer.save_pack(str(run_dir / "tokenizer.nsos"))
        eos_token_id = tokenizer.encode("<|endoftext|>")[0]

        if args.device == "gpu":
            device = nsos.Device.GPU
        elif args.device == "cpu":
            device = nsos.Device.CPU
        else:
            device = nsos.Device.GPU if os.name == "nt" else nsos.Device.CPU

        if device == nsos.Device.GPU and hasattr(nsos, "fast_gpu_supported"):
            if not nsos.fast_gpu_supported():
                logger.log(
                    "[device] Fast GPU path unavailable on this CUDA/toolkit/GPU combination; "
                    "falling back to CPU for correctness and throughput."
                )
                device = nsos.Device.CPU

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
        logger.log(
            f"[profile] requested={args.profile} canonical={canonical_profile} "
            f"family={effective_schedule['profile_family']} dominant={effective_schedule['dominant_stack']}"
        )
        logger.log(f"[profile] scope={effective_schedule['validation_scope']}")

        model = nsos.JambaModel(model_config, device)
        model.to(device)
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
        trainer = nsos.Trainer(model, profile["lr"])
        trainer.weight_decay = profile["weight_decay"]
        trainer.max_grad_norm = profile["max_grad_norm"]
        trainer.warmup_steps = profile["warmup_steps"]
        trainer.min_learning_rate_scale = profile["min_lr_scale"]
        trainer.first_token_loss_scale = profile.get("first_token_loss_scale", 2.5)
        trainer.eos_loss_scale = profile.get("eos_loss_scale", 0.35)
        trainer.repetition_unlikelihood_scale = phase_repetition_scale(
            profile,
            "phase1_algorithms",
        )
        trainer.eos_token_id = eos_token_id
        trainer.total_training_steps = (
            sum(args.override_phase_steps or profile["phase_steps"][phase] for phase in PHASE_ORDER)
            + int(profile.get("instruction_polish_steps", 0))
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
            "global_suite_samples_per_phase": args.global_suite_samples_per_phase,
            "global_suite_exact_samples": args.global_suite_exact_samples,
            "replay_ratio": args.replay_ratio,
            "final_consolidation_steps": args.final_consolidation_steps,
            "phase_best_eval_every_steps": args.phase_best_eval_every_steps,
            "log_ema_beta": args.log_ema_beta,
            "progress_mode": logger.mode,
            "total_training_steps": trainer.total_training_steps,
            "generation_guard": {
                "repetition_penalty": GENERATION_REPETITION_PENALTY,
                "no_repeat_ngram_size": GENERATION_NO_REPEAT_NGRAM,
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

        metrics_path = run_dir / "metrics.jsonl"
        supervised_history: List[Dict] = []
        supervised_history_by_family: Dict[str, List[Dict]] = {}
        replay_rng = random.Random(args.seed + 9001)
        base_learning_rate = float(profile["lr"])
        global_suite = build_global_suite(args.bundle_dir, args.global_suite_samples_per_phase)
        best_global_score = float("-inf")
        best_release_score = float("-inf")
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            for phase_index, phase_name in enumerate(PHASE_ORDER):
                train_rows = curriculum_texts_for_phase(args.bundle_dir, phase_name, "train")
                eval_rows = curriculum_texts_for_phase(args.bundle_dir, phase_name, "eval")
                train_tokens = build_token_stream(tokenizer, train_rows, "<|endoftext|>")
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
                if phase_name != "phase3_curated_text":
                    phase_rows.extend(
                        sample_phase_replay_rows(
                            phase_name,
                            supervised_history_by_family,
                            args.replay_ratio,
                            len(train_rows),
                            replay_rng,
                        )
                    )

                logger.log(
                    f"[train] {phase_name}: samples={len(train_rows)} mixed={len(phase_rows)} "
                    f"tokens={len(train_tokens)} steps={max_steps} seq_len={profile['seq_len']} "
                    f"batch={profile['batch_size']} lr={trainer.learning_rate:.2e} "
                    f"rul={trainer.repetition_unlikelihood_scale:.3f} "
                    f"aux={{enabled:{int(phase_aux_cfg['enabled'])},"
                    f"ttt:{int(phase_aux_cfg['session_adapt'])},"
                    f"reason:{int(phase_aux_cfg['reasoning'])},"
                    f"mem:{int(phase_aux_cfg['memory'])}}}"
                )
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

                    if args.checkpoint_every_steps > 0 and step % args.checkpoint_every_steps == 0:
                        save_checkpoint_artifact(f"{phase_name}_step{step}")
                    if (
                        args.phase_best_eval_every_steps > 0
                        and eval_subset
                        and (step % args.phase_best_eval_every_steps == 0 or step == max_steps)
                    ):
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
                    supervised_history.extend(train_rows)
                    family = PHASE_FAMILIES.get(phase_name)
                    if family:
                        supervised_history_by_family.setdefault(family, []).extend(train_rows)

                selected_source = "final"
                if best_phase_name:
                    debug_eval_log(logger, f"{phase_name}:loading_best:start {best_phase_name}")
                    model.load(str(run_dir / f"{best_phase_name}.bin"))
                    debug_eval_log(logger, f"{phase_name}:loading_best:done {best_phase_name}")
                    save_checkpoint_artifact(f"{phase_name}_best")
                    selected_source = "best_eval"
                    phase_metrics = (
                        dict(best_phase_metrics)
                        if best_phase_metrics is not None
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
                summary["phases"].append(phase_summary)
                metrics_file.write(json.dumps(phase_summary, ensure_ascii=False) + "\n")
                metrics_file.flush()

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
                    polish_metrics = dict(polish_best_metrics) if polish_best_metrics is not None else evaluate_phase(
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
                summary["phases"].append(polish_summary)
                metrics_file.write(json.dumps(polish_summary, ensure_ascii=False) + "\n")
                metrics_file.flush()
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
                for phase_name in PHASE_ORDER:
                    if phase_name == "phase3_curated_text":
                        continue
                    consolidation_eval_rows.extend(
                        curriculum_texts_for_phase(args.bundle_dir, phase_name, "eval")[
                            : max(1, args.phase_eval_samples // 2)
                        ]
                    )
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
                summary["phases"].append(consolidation_summary)
                metrics_file.write(json.dumps(consolidation_summary, ensure_ascii=False) + "\n")
                metrics_file.flush()
                save_checkpoint_artifact("final_consolidation")
                logger.log(
                    f"[eval] final_consolidation: answer_loss={consolidation_summary['answer_loss']:.4f} "
                    f"first={consolidation_summary['first_token_accuracy']:.2f} "
                    f"teacher={consolidation_summary['teacher_token_accuracy']:.2f}"
                )

        final_model_source = "current"
        release_candidate_path = run_dir / "release_candidate.bin"
        champion_path = run_dir / "champion_global.bin"
        if summary.get("release_candidate", {}).get("source") and release_candidate_path.exists():
            model.load(str(release_candidate_path))
            final_model_source = "release_candidate"
        elif summary.get("global_champion", {}).get("source") and champion_path.exists():
            model.load(str(champion_path))
            final_model_source = "champion_global"
        summary["final_model_source"] = final_model_source
        model.save(str(run_dir / "final_model.bin"))
        model.save_edge_linear_pack(str(run_dir / "final_edge_linear.nsos"))
        save_run_summary(run_dir / "run_summary.json", summary)
        logger.log(f"[done] final checkpoint: {run_dir / 'final_model.bin'}")
        logger.log(f"[done] final edge pack: {run_dir / 'final_edge_linear.nsos'}")
        logger.log(f"[done] summary: {run_dir / 'run_summary.json'}")
        return 0
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
