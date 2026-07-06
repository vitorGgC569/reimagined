import json
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
    for directory in candidates:
        if directory.is_dir():
            _dll_handles.append(os.add_dll_directory(str(directory)))

import nsos_ext


def main() -> int:
    config = nsos_ext.ModelConfig()
    config.num_layers = 1
    config.d_model = 32
    config.vocab_size = 256
    config.max_context_tokens = 64

    engine = nsos_ext.InferenceEngine()
    assert engine.load_model("", config)
    text = engine.generate("hi", 1, 0.0)
    assert isinstance(text, str)
    json.dumps({"text": text})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
