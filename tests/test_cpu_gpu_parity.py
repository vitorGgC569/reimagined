"""Fail-closed Python CPU/GPU model parity.

This incubation test is not a supported-product release gate, but it must never
report success without executing CUDA. Set NSOS_BUILD_DIR to the directory that
contains the CUDA-enabled ``nsos_ext`` module.
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile

import numpy as np
import torch


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
BUILD_DIR = pathlib.Path(
    os.environ.get("NSOS_BUILD_DIR", ROOT_DIR / "OXN" / "nsos" / "build")
).resolve()
sys.path.insert(0, str(BUILD_DIR))

try:
    import nsos_ext
except ImportError as error:
    raise RuntimeError(
        f"CUDA parity requires nsos_ext in NSOS_BUILD_DIR={BUILD_DIR}"
    ) from error


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for test_cpu_gpu_parity")
    if not hasattr(nsos_ext, "Device") or not hasattr(nsos_ext.Device, "GPU"):
        raise RuntimeError("nsos_ext was built without the GPU Device binding")


def test_cpu_gpu_parity() -> None:
    require_cuda()
    nsos_ext.set_strict_gpu_execution(True)
    nsos_ext.set_seed(42)

    model_cpu = nsos_ext.JambaModel(1, 64, 100, nsos_ext.Device.CPU)
    model_cpu.set_training_mode(False)
    input_ids = [1, 2, 3, 4]

    checkpoint_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="nsos_python_gpu_parity_",
            suffix=".bin",
            delete=False,
        ) as checkpoint:
            checkpoint_path = pathlib.Path(checkpoint.name)
        model_cpu.save(str(checkpoint_path))

        model_gpu = nsos_ext.JambaModel(1, 64, 100, nsos_ext.Device.GPU)
        model_gpu.load(str(checkpoint_path), True)
        model_gpu.to(nsos_ext.Device.GPU)
        model_gpu.set_training_mode(False)

        output_cpu = np.asarray(model_cpu.forward_ids(input_ids).numpy()).copy()
        output_gpu_tensor = model_gpu.forward_ids(input_ids)
        if output_gpu_tensor.device != nsos_ext.Device.GPU:
            raise AssertionError("GPU model returned a non-GPU Tensor")
        output_gpu = np.asarray(output_gpu_tensor.cpu().numpy()).copy()

        np.testing.assert_allclose(
            output_cpu,
            output_gpu,
            rtol=5e-3,
            atol=5e-3,
            err_msg="CPU/GPU Jamba logits diverged",
        )
    finally:
        if checkpoint_path is not None:
            checkpoint_path.unlink(missing_ok=True)


if __name__ == "__main__":
    test_cpu_gpu_parity()
    print("CPU/GPU parity PASS")
