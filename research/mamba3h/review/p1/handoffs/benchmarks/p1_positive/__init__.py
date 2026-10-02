"""Fresh P1 positive MQAR; frozen P0 modules/manifests are not edited."""

import os

for _thread_flag in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_flag] = "2"

from .generation import generate, build_corpus
from .oracle import solve
from .schema import model_view, model_numeric, encode_numeric

__all__ = ["generate", "build_corpus", "solve", "model_view", "model_numeric", "encode_numeric"]
