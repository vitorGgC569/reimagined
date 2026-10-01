"""Fail-closed Python GPU memory-contract smoke test."""

from __future__ import annotations

import os
import pathlib
import sys

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
        f"GPU memory test requires nsos_ext in NSOS_BUILD_DIR={BUILD_DIR}"
    ) from error


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for test_gpu_safe")
    if not hasattr(nsos_ext, "Device") or not hasattr(nsos_ext.Device, "GPU"):
        raise RuntimeError("nsos_ext was built without the GPU Device binding")


def test_cpu_copy_contract() -> None:
    tensor = nsos_ext.Tensor([2, 2], nsos_ext.Device.CPU, 3.0)
    np.testing.assert_array_equal(tensor.numpy(), np.full((2, 2), 3.0))
    assert tensor.cpu().device == nsos_ext.Device.CPU


def test_gpu_device_memory_contract() -> None:
    require_cuda()
    nsos_ext.set_strict_gpu_execution(True)

    gpu_tensor = nsos_ext.Tensor([2, 2], nsos_ext.Device.GPU, 3.0)
    assert gpu_tensor.device == nsos_ext.Device.GPU

    gpu_result = gpu_tensor.add(gpu_tensor)
    assert gpu_result.device == nsos_ext.Device.GPU
    host_result = np.asarray(gpu_result.cpu().numpy()).copy()
    np.testing.assert_array_equal(host_result, np.full((2, 2), 6.0))


if __name__ == "__main__":
    test_cpu_copy_contract()
    test_gpu_device_memory_contract()
    print("GPU memory contract PASS")
