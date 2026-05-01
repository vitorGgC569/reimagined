"""
Experimental Python blueprint for NSOS evaluation ideas.

The production evaluation path lives in the native test/eval tooling and in
OXN/nsos/scripts/train_curriculum.py.
"""

EXPERIMENTAL_BLUEPRINT_MESSAGE = (
    "OXN/nsos/evaluation.py is an experimental blueprint, not the release "
    "evaluation harness. Use the native tests and curriculum evaluation tools."
)


class NSOSEvaluator:
    def __init__(self, model, config):
        raise RuntimeError(EXPERIMENTAL_BLUEPRINT_MESSAGE)

    def evaluate(self, benchmark_suite):
        raise RuntimeError(EXPERIMENTAL_BLUEPRINT_MESSAGE)

    def run_needle_in_a_haystack(self, lengths=None):
        raise RuntimeError(EXPERIMENTAL_BLUEPRINT_MESSAGE)
