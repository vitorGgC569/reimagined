"""
Experimental Python blueprint for NSOS data pipeline ideas.

The production-ready training data path is the native/runtime-backed curriculum
pipeline under OXN/nsos/scripts/.
"""

EXPERIMENTAL_BLUEPRINT_MESSAGE = (
    "OXN/nsos/data_pipeline.py is an experimental blueprint, not the shipping "
    "data pipeline. Use OXN/nsos/scripts/fetch_real_datasets.py and "
    "OXN/nsos/scripts/train_curriculum.py instead."
)


class QualityFilter:
    def __init__(self, min_len=100, max_len=100000):
        raise RuntimeError(EXPERIMENTAL_BLUEPRINT_MESSAGE)


class Deduplication:
    def __init__(self):
        raise RuntimeError(EXPERIMENTAL_BLUEPRINT_MESSAGE)


class NSOSDataPipeline:
    def __init__(self, config):
        raise RuntimeError(EXPERIMENTAL_BLUEPRINT_MESSAGE)

    def stream_dataset(self, name, split="train", limit=None):
        raise RuntimeError(EXPERIMENTAL_BLUEPRINT_MESSAGE)

    def process_batch(self, batch):
        raise RuntimeError(EXPERIMENTAL_BLUEPRINT_MESSAGE)

    def get_phase_1_loader(self):
        raise RuntimeError(EXPERIMENTAL_BLUEPRINT_MESSAGE)
