"""Create an UNTRAINED synthetic runtime fixture, never a model-quality benchmark.

Use measure_decode_repro.py with the same saved artifacts across binaries.
The byte tokenizer is deliberately artificial. No corpus or training is performed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from cuda_env import add_windows_runtime_dirs

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    build = args.build_dir.resolve(strict=True)
    add_windows_runtime_dirs(build)
    sys.path.insert(0, str(build))
    import nsos_ext as nsos
    if not Path(nsos.__file__).resolve().is_relative_to(build):
        raise RuntimeError("Wrong extension loaded")
    nsos.set_seed(20260919)
    tokenizer = nsos.Tokenizer()
    tokenizer.add_special_tokens(["<|synthetic_eos|>"])
    eos = list(tokenizer.encode("<|synthetic_eos|>"))
    if len(eos) != 1:
        raise RuntimeError("Synthetic EOS must be exactly one token")
    tokenizer.save_pack(str(directory / "tokenizer.nsos"))
    configuration = {
        "num_layers": 6, "d_model": 256, "vocab_size": tokenizer.vocab_size,
        "n_heads": 8, "n_kv_heads": 2, "attention_period": 2, "attention_slot": 1,
        "mamba_d_state": 32, "mamba_head_dim": 32, "mamba_n_groups": 1,
        "mamba2_faithful": True, "use_moe": True, "num_experts": 4,
        "num_experts_per_token": 2, "moe_period": 2, "moe_slot": 0,
        "moe_expert_hidden_dim": 512, "use_ttt": False, "use_chrass": False,
        "use_kan": False, "dropout": 0.0, "max_context_tokens": 512,
        "sliding_window": 512, "use_cuda": False,
    }
    cfg = nsos.ModelConfig()
    for key, value in configuration.items():
        setattr(cfg, key, value)
    engine = nsos.InferenceEngine()
    if not engine.load_model("", cfg):
        raise RuntimeError("Synthetic model construction failed")
    if not engine.save_checkpoint(str(directory / "UNTRAINED.bin")):
        raise RuntimeError("Synthetic checkpoint save failed")
    for name, value in (
        ("config.json", configuration),
        ("prompts.json", ["synthetic runtime probe: 0123456789", "synthetic runtime probe: 9876543210"]),
        ("fixture.json", {"kind": "UNTRAINED_SYNTHETIC_RUNTIME_ONLY", "quality_validated": False,
                          "eos_token_id": eos[0], "parameter_count": engine.parameter_count(), "seed": 20260919}),
    ):
        with (directory / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
    print(json.dumps({"fixture": str(directory), "eos_token_id": eos[0], "parameters": engine.parameter_count()}))

if __name__ == "__main__":
    main()
