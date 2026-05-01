import sys
import os
import time
import numpy as np

# Add build path
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))

try:
    import nsos_ext
except ImportError:
    print("Error: nsos_ext not found.")
    sys.exit(1)

print("=== NSOS Reasoning Benchmark (System 2) ===")

# 1. Setup
d_model = 64
# For simulation, we assume MCTS search iterates and 'evaluates'.
# The MCTS evaluation in C++ uses a stub that checks if latent sum > 10.
# We want to measure Time-to-Solution.

def run_proof_search(difficulty="short"):
    # Initialize MCTS with a state
    # If difficulty is long, we might need more iterations to find a "winning" node
    # But our MCTS simulation is stochastic/stubbed.
    # We measure overhead of the search loop.

    initial_state = nsos_ext.Tensor([1, d_model], nsos_ext.Device.CPU, 0.0)
    # Inject signal to guide search?
    # Our C++ MCTS expands using JambaModel.
    # We need a JambaModel instance.

    # Python bindings for MCTS constructor take Tensor.
    # The JambaModel binding inside MCTS wasn't fully exposed to Python construction in bindings.cpp
    # (MCTS constructor only takes Tensor in Python binding).
    # So MCTS uses nullptr model -> Identity/Random expansion.

    mcts = nsos_ext.MCTS(initial_state)

    iterations = 50 if difficulty == "short" else 200

    start = time.time()
    mcts.search(iterations)
    end = time.time()

    best = mcts.get_best_action()
    print(f"[{difficulty.upper()}] Iterations: {iterations} | Time: {end-start:.4f}s | Best Action Token: {best}")

run_proof_search("short")
run_proof_search("long")
