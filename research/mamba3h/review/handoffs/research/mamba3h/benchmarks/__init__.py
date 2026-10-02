"""Independent CPU-only synthetic benchmark contracts (not a model baseline)."""

import os

# Must precede any optional numerical-library import in this package.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

from .generation import generate
from .oracle import solve
from .schema import model_view, encode_numeric, compact_targets, label_vocabulary

__all__ = ["generate", "solve", "model_view", "encode_numeric", "compact_targets", "label_vocabulary"]
