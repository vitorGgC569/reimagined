import os
import sys
import torch
import numpy as np

# Force Determinism
os.environ["OMP_NUM_THREADS"] = "1"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(ROOT_DIR, "build"))

try:
    import nsos_ext
except ImportError:
    sys.exit(1)

def test_save_load_equivalence():
    print("🔬 Testing Save/Load Equivalence (KAN Reset Verifier)...")

    DIM = 64
    LAYERS = 2
    nsos_ext.set_seed(42)

    # Model A
    model_a = nsos_ext.JambaModel(LAYERS, DIM, 128)
    ctx_a = nsos_ext.Context()
    input_ids = [1, 5, 9, 2]

    out_a = model_a.forward_ids(input_ids, ctx_a)
    val_a = np.array(out_a, copy=True)

    model_a.save("debug_model.bin")

    # Model B
    model_b = nsos_ext.JambaModel(LAYERS, DIM, 128)
    model_b.load("debug_model.bin")

    # CRITICAL: Call Reset
    # This should reset KAN RNG to seed 12345.
    model_b.reset_session()

    # Forward B
    ctx_b = nsos_ext.Context()
    # Ensure Global RNG matches Model A start?
    # Model A used global RNG for TTT noise.
    # Model B uses global RNG for TTT noise.
    # TTT layer has its OWN RNG, but we updated it to use default seed?
    # No, TTT layer uses member `rng` initialized with 12345 in constructor.
    # And reset() sets it to 12345.

    # BUT wait: TTT Layer constructor `rng(12345)` vs `reset()` `rng = Xoroshiro(12345)`.
    # This seems correct.

    # What about Global RNG for other things?
    # Embeddings? Forward pass doesn't use random unless dropout.
    # KAN uses random rounding.

    # Is it possible Model A KAN was initialized with `nsos::get_rng()()` so it had a random seed?
    # Model A Constructor:
    # `gen = std::mt19937(nsos::get_rng()());`
    # `nsos::get_rng()` is seeded with 42 via `set_seed(42)`.
    # So `gen` in A gets a deterministic seed derived from 42.

    # Model B Constructor:
    # `gen = std::mt19937(nsos::get_rng()());`
    # This consumes more from global RNG (since A used some).

    # Model B Reset:
    # `gen.seed(12345);`

    # AHA! Model A's KAN `gen` was seeded with something from `nsos::get_rng()`, NOT 12345.
    # Model B's KAN `gen` is reset to 12345.
    # They are DIFFERENT.

    # To match, Model A must ALSO be reset before use, OR Model A must be initialized with 12345?
    # The requirement is "Deterministic after reset".
    # So we should call `model_a.reset_session()` BEFORE the first forward too?
    # YES. This puts A in the canonical "Reset" state.

    print("   Resetting Model A before run...")
    model_a.reset_session()
    out_a_reset = model_a.forward_ids(input_ids, ctx_a)
    val_a_reset = np.array(out_a_reset, copy=True)

    # Now compare B (already reset) to A (reset)

    # Re-run B
    model_b.reset_session() # Just to be sure
    out_b = model_b.forward_ids(input_ids, ctx_b)
    val_b = np.array(out_b, copy=True)

    diff = np.max(np.abs(val_a_reset - val_b))
    print(f"   Output Diff (A_reset vs B_reset): {diff}")

    if diff > 1e-4:
        print("FAILURE")
    else:
        print("SUCCESS")

    if os.path.exists("debug_model.bin"): os.remove("debug_model.bin")

if __name__ == "__main__":
    test_save_load_equivalence()
