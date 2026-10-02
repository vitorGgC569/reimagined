import json
import numpy as np
import os
from pathlib import Path


_dll_handles = []
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    candidates = []
    for key in ("CUDA_PATH", "CUDA_PATH_V12_9"):
        if os.environ.get(key):
            candidates.append(Path(os.environ[key]) / "bin")
    candidates.extend(
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA").glob(
            "v*/bin"
        )
    )
    for key in ("NSOS_HIP_ROOT", "ROCM_PATH", "HIP_PATH"):
        if os.environ.get(key):
            root = Path(os.environ[key])
            candidates.extend((root / "bin", root / "lib" / "llvm" / "bin"))
    default_hip_root = Path(r"C:\TheRock\build")
    candidates.extend(
        (default_hip_root / "bin", default_hip_root / "lib" / "llvm" / "bin")
    )
    for directory in candidates:
        if directory.is_dir():
            _dll_handles.append(os.add_dll_directory(str(directory)))

import nsos_ext


def check_empty(t, expected):
    n = nsos_ext
    assert t.size==0
    assert t.clone().size==0, "empty clone invented scalar storage"
    for copy in (t,t.clone(),t.cpu().clone(),t.to(n.Device.GPU).cpu().clone()):
        assert copy.size==0 and copy.shape==t.shape
        a=copy.numpy()
        assert a.shape==expected and a.size==0 and a.dtype==np.float32
        assert np.isfinite(a).all()


def empty_binding_contract():
    n = nsos_ext
    linear=n.BitLinear(4,3,True)
    parameters=linear.parameters()
    assert parameters and all(p.grad.size==0 for p in parameters)
    for p in parameters: check_empty(p.grad,(0,))
    for dims in ([0],[2,0,3],[0,4],[3,2,0],[0,0]):
        for device in (n.Device.CPU,n.Device.GPU):check_empty(n.Tensor(dims,device),tuple(dims))
    scalar=n.Tensor([],n.Device.CPU,1.25)
    assert scalar.size==1 and scalar.shape==[]
    for t in (scalar,scalar.clone()):
        a=t.numpy();assert a.shape==() and a.size==1 and float(a)==1.25
    source=n.Tensor.ones([2,3],n.Device.CPU)
    array=source.numpy()
    assert array.shape==(2,3) and np.array_equal(array,np.ones((2,3),np.float32))
    array[0,0]=9
    assert source.numpy()[0,0]==1
    del source
    assert array[0,0]==9

def main() -> int:
    empty_binding_contract()
    backend = nsos_ext.gpu_backend_name()
    vendor = nsos_ext.gpu_vendor_name()
    assert backend in {"none", "cuda", "hip"}
    assert vendor in {"none", "nvidia", "amd"}
    devices = list(nsos_ext.gpu_devices())
    if backend == "none":
        assert vendor == "none"
        assert devices == []
        expected_selection_error = False
        try:
            nsos_ext.selected_gpu_device()
        except RuntimeError:
            expected_selection_error = True
        else:
            raise AssertionError(
                "CPU-only binding unexpectedly selected a GPU"
            )
        assert expected_selection_error
    else:
        assert vendor == ("amd" if backend == "hip" else "nvidia")
        selected = int(nsos_ext.selected_gpu_device())
        matches = [
            device
            for device in devices
            if int(device["index"]) == selected
        ]
        assert len(matches) == 1
        assert bool(matches[0]["compiled"])

    benchmark_input = nsos_ext.Tensor.ones([2, 4], nsos_ext.Device.CPU)
    linear = nsos_ext.BitLinear(4, 3, True)
    assert linear.forward(benchmark_input).shape == [2, 3]
    assert len(linear.parameters()) > 0
    mamba = nsos_ext.Mamba2SSD(4, 2, 1)
    assert mamba.forward(benchmark_input, None).shape == [2, 4]
    router = nsos_ext.MoERouter(4, 2, 1)
    logits, weights = router.forward(benchmark_input)
    assert logits.shape == [2, 2]
    assert weights.shape == [2, 2]

    config = nsos_ext.ModelConfig()
    config.num_layers = 1
    config.d_model = 32
    config.vocab_size = 256
    config.n_heads = 4
    config.n_kv_heads = 2
    config.attention_period = 64
    config.attention_slot = 63
    config.use_moe = False
    config.use_ttt = False
    config.max_context_tokens = 64

    engine = nsos_ext.InferenceEngine()
    assert engine.load_model("", config)
    engine.request_training_cancellation()
    assert engine.training_cancellation_requested()
    engine.clear_training_cancellation()
    assert not engine.training_cancellation_requested()
    assert not engine.training_optimizer_state_poisoned()
    text = engine.generate("hi", 1, 0.0)
    assert isinstance(text, str)
    json.dumps({"text": text})

    training_model = nsos_ext.JambaModel(config, nsos_ext.Device.CPU)
    state = nsos_ext.Tensor.zeros([32], nsos_ext.Device.CPU)
    try:
        training_model.reason(state, 4)
    except RuntimeError:
        pass
    else:
        raise AssertionError("unverified reasoning was accepted")
    policy = nsos_ext.ReasoningPolicy()
    policy.policy_id = "binding:ones:v1"
    policy.verifier_id = "binding:exact-first-token:v1"
    def propose(state, depth, limit):
        item = nsos_ext.ReasoningProposal()
        item.state = nsos_ext.Tensor.ones([32], nsos_ext.Device.CPU)
        item.prior = 1.0
        return [item] if limit else []
    policy.propose = propose
    policy.verify = nsos_ext.exact_token_verifier([1], lambda state: [int(state.numpy()[0])])
    training_model.set_reasoning_policy(policy)
    assert training_model.has_reasoning_policy()
    refined = training_model.reason(state, 4)
    assert int(refined.numpy()[0]) == 1
    assert training_model.last_reasoning_report().best_score == 1.0
    assert training_model.last_reasoning_report().evaluated_states == 2
    training_model.clear_reasoning_policy()
    assert not training_model.has_reasoning_policy()
    trainer = nsos_ext.Trainer(training_model, 1.0e-3)
    assert not trainer.cancellation_requested()
    trainer.request_cancellation()
    assert trainer.cancellation_requested()
    trainer.clear_cancellation()
    assert not trainer.cancellation_requested()
    assert not trainer.optimizer_state_poisoned()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
