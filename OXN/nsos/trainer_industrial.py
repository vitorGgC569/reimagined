"""
Experimental Python blueprint for large-scale NSOS training ideas.

The production training path lives in OXN/nsos/scripts/train_curriculum.py and
the native C++ trainer/runtime.
"""

EXPERIMENTAL_BLUEPRINT_MESSAGE = (
    "OXN/nsos/trainer_industrial.py is an experimental blueprint, not the "
    "shipping trainer. Use OXN/nsos/scripts/train_curriculum.py and the native "
    "C++ trainer instead."
)


class NSOSTrainerIndustrial:
    def __init__(self, model, config):
        raise RuntimeError(EXPERIMENTAL_BLUEPRINT_MESSAGE)

    def run_pretraining(self, train_loader):
        raise RuntimeError(EXPERIMENTAL_BLUEPRINT_MESSAGE)

    def run_sft(self, sft_dataset):
        raise RuntimeError(EXPERIMENTAL_BLUEPRINT_MESSAGE)

    def run_orpo(self, preference_dataset):
        raise RuntimeError(EXPERIMENTAL_BLUEPRINT_MESSAGE)

    def run_ttt_meta_learning(self, meta_dataset):
        raise RuntimeError(EXPERIMENTAL_BLUEPRINT_MESSAGE)

    def save_checkpoint(self, path):
        raise RuntimeError(EXPERIMENTAL_BLUEPRINT_MESSAGE)


if __name__ == "__main__":
    raise SystemExit(EXPERIMENTAL_BLUEPRINT_MESSAGE)
