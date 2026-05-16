from pathlib import Path
import sys
import numpy as np


ROOT = Path(__file__).resolve().parent
BUILD_DIRS = [
    ROOT / "OXN" / "nsos" / "build_nmake_release",
    ROOT / "OXN" / "nsos" / "build" / "Release",
    ROOT / "OXN" / "nsos" / "build",
]

for build_dir in BUILD_DIRS:
    if build_dir.exists():
        sys.path.insert(0, str(build_dir))

import nsos_ext as nsos  # noqa: E402


def generate(model, prompt_text: str, max_tokens: int = 150) -> str:
    generated = [ord(c) for c in prompt_text]
    for _ in range(max_tokens):
        ctx = nsos.Context()
        logits = model.forward_ids(generated, ctx)
        logits_np = logits.numpy()
        next_id = int(np.argmax(logits_np[-1]))
        if next_id in (0, ord("\n")):
            break
        generated.append(next_id)
    return "".join(chr(i) for i in generated)


def infer() -> None:
    config = {"layers": 2, "dim": 128, "vocab": 256}
    model_path = ROOT / "circuit_brain_checkpoint.bin"
    if not model_path.exists():
        raise FileNotFoundError(
            "Missing circuit_brain_checkpoint.bin. Train the model first."
        )

    print("=" * 80)
    print("CIRCUIT-OXN INFERENCE ENGINE")
    print("=" * 80)

    model = nsos.JambaModel(
        config["layers"], config["dim"], config["vocab"], nsos.Device.CPU
    )
    model.load(str(model_path))

    prompts = [
        "REQ:voltage_divider_10_20|NET:",
        "REQ:rc_filter_5k_10n|NET:",
        "REQ:led_driver_9v|NET:",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print(f"Result: {generate(model, prompt)}")


if __name__ == "__main__":
    infer()
