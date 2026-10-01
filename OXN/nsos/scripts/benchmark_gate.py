from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from native_module import load_native_module, native_artifact_identity, resolve_native_build_dir

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight NSOS runtime benchmark gate.")
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--profile", choices=["mamba_small", "hybrid_pilot"], default="mamba_small")
    parser.add_argument("--max-tokens", type=int, default=24)
    # Defaults are the thresholds PRODUCT.md declares for the release gate.
    # They used to be 0.0, which silently turned the throughput budget into a
    # no-op for every invocation that did not pass the flags explicitly --
    # including the one documented in docs/RELEASE.md. Lower them per run only
    # with a recorded reason.
    parser.add_argument("--min-prompt-tok-s", type=float, default=1.0)
    parser.add_argument("--min-decode-tok-s", type=float, default=0.1)
    parser.add_argument("--max-sampler-share", type=float, default=1.0)
    parser.add_argument("--max-memory-bytes", type=int, default=0)
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args()


def detect_build_dir(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    repo_root = Path(__file__).resolve().parents[3]
    nsos_root = repo_root / "OXN" / "nsos"
    for name in ["build-mvp", "build_cuda129", "build_v1", "build_full", "build_codex", "build-ci-local", "build"]:
        candidates.extend([nsos_root / name / "Release", nsos_root / name])
    return resolve_native_build_dir(explicit, candidates)


def load_nsos(build_dir: Path):
    return load_native_module(build_dir)


def build_config(nsos, profile_name: str, device):
    config = nsos.ModelConfig()
    if profile_name == "mamba_small":
        config.num_layers = 4
        config.d_model = 96
        config.vocab_size = 320
        config.n_heads = 4
        config.n_kv_heads = 2
        config.attention_period = 64
        config.attention_slot = 63
        config.use_moe = False
        config.use_ttt = False
        config.use_exact_attention_training = False
    else:
        config.num_layers = 6
        config.d_model = 128
        config.vocab_size = 320
        config.n_heads = 4
        config.n_kv_heads = 2
        config.attention_period = 2
        config.attention_slot = 1
        config.use_moe = False
        config.use_ttt = False
        config.use_exact_attention_training = True
    config.max_context_tokens = 256
    config.default_batch_size = 2
    config.use_cuda = device == nsos.Device.GPU
    return config


def metrics_to_dict(metrics) -> dict:
    return {
        "prompt_tokens_total": int(metrics.prompt_tokens_total),
        "prompt_tokens_used": int(metrics.prompt_tokens_used),
        "generated_tokens": int(metrics.generated_tokens),
        "batch_size": int(metrics.batch_size),
        "elapsed_ms": float(metrics.elapsed_ms),
        "prefill_ms": float(getattr(metrics, "prefill_ms", 0.0)),
        "decode_ms": float(getattr(metrics, "decode_ms", 0.0)),
        "sampler_ms": float(metrics.sampler_ms),
        "prompt_tokens_per_sec": float(metrics.prompt_tokens_per_sec),
        "decode_tokens_per_sec": float(metrics.decode_tokens_per_sec),
        "total_tokens_per_sec": float(getattr(metrics, "total_tokens_per_sec", 0.0)),
        "used_streaming": bool(metrics.used_streaming),
        "loaded_from_pack": bool(metrics.loaded_from_pack),
        "mamba_fast_path_hits": int(metrics.mamba_fast_path_hits),
        "mamba_fast_path_fallbacks": int(metrics.mamba_fast_path_fallbacks),
        "mamba_last_fallback_reason": str(metrics.mamba_last_fallback_reason),
    }


def assert_budget(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    args = parse_args()
    build_dir = detect_build_dir(args.build_dir)
    nsos = load_nsos(build_dir)
    device = nsos.Device.GPU if args.device == "gpu" else nsos.Device.CPU
    config = build_config(nsos, args.profile, device)

    prompt = (
        "NSOS industrial benchmark prompt. "
        "Explain why compact weights, deterministic reload, and fast streaming matter on edge serving. "
        "List two operational risks of host-side fallback."
    )

    with tempfile.TemporaryDirectory(prefix="nsos-bench-") as tmp:
        tmpdir = Path(tmp)
        checkpoint_path = tmpdir / "model.bin"
        tokenizer_path = tmpdir / "tokenizer.nsos"
        pack_dir = tmpdir / "pack"

        tokenizer = nsos.Tokenizer()
        tokenizer.add_special_tokens(["<|endoftext|>"])
        tokenizer.save_pack(str(tokenizer_path))

        model = nsos.JambaModel(config, device)
        model.to(device)
        model.save(str(checkpoint_path))

        engine = nsos.InferenceEngine()
        if not engine.load_model(str(checkpoint_path), config):
            raise RuntimeError("Engine failed to load synthetic checkpoint")

        options = nsos.GenerationOptions()
        options.max_tokens = args.max_tokens
        options.min_new_tokens = min(8, args.max_tokens)
        options.temperature = 0.2
        options.top_p = 0.9
        options.top_k = 1
        options.max_context_tokens = config.max_context_tokens
        options.eos_token_id = tokenizer.encode("<|endoftext|>")[0]

        first_output = engine.generate_ex(prompt, options)
        first_metrics = metrics_to_dict(engine.last_generation_metrics())
        first_memory = int(engine.get_memory_usage())

        if not engine.save_model_pack(str(pack_dir)):
            raise RuntimeError("Engine failed to save model pack during benchmark gate")

        reloaded_engine = nsos.InferenceEngine()
        if not reloaded_engine.load_model(str(pack_dir), config):
            raise RuntimeError("Engine failed to reload saved model pack")
        second_output = reloaded_engine.generate_ex(prompt, options)
        second_metrics = metrics_to_dict(reloaded_engine.last_generation_metrics())
        second_memory = int(reloaded_engine.get_memory_usage())

        sampler_share = first_metrics["sampler_ms"] / max(first_metrics["elapsed_ms"], 1e-9)
        reload_deterministic = first_output == second_output

        report = {
            "build_dir": str(build_dir),
            "native_artifact": native_artifact_identity(nsos),
            "device": args.device,
            "profile": args.profile,
            "metrics": first_metrics,
            "reloaded_metrics": second_metrics,
            "memory_bytes": first_memory,
            "reloaded_memory_bytes": second_memory,
            "sampler_share": sampler_share,
            "reload_deterministic": reload_deterministic,
            "first_output": first_output,
            "reloaded_output": second_output,
        }

        assert_budget(
            first_metrics["prompt_tokens_per_sec"] >= args.min_prompt_tok_s,
            f"prompt tok/s below budget: {first_metrics['prompt_tokens_per_sec']:.2f} < {args.min_prompt_tok_s:.2f}",
        )
        assert_budget(
            first_metrics["decode_tokens_per_sec"] >= args.min_decode_tok_s,
            f"decode tok/s below budget: {first_metrics['decode_tokens_per_sec']:.2f} < {args.min_decode_tok_s:.2f}",
        )
        assert_budget(
            sampler_share <= args.max_sampler_share,
            f"sampler share above budget: {sampler_share:.3f} > {args.max_sampler_share:.3f}",
        )
        if args.max_memory_bytes > 0:
            assert_budget(
                first_memory <= args.max_memory_bytes,
                f"memory above budget: {first_memory} > {args.max_memory_bytes}",
            )
        assert_budget(reload_deterministic, "model reload is not deterministic under benchmark gate")

        report_path = args.report_path or (Path(build_dir) / "benchmark_gate.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=True))
        print(f"[done] benchmark gate report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
