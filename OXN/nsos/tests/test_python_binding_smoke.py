import json

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
