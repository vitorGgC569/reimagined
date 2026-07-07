# NSOS Mamba standalone module

Status: experimental module wrapper over `nsos_ext`.  It does not change the
Mamba architecture.  It only builds the faithful Mamba-2 NSOS configuration,
selects CPU/GPU, enables streaming incremental decoding by default, and
standardizes small supervised datasets.

## Goals

- expose only the Mamba path, without Attention, MoE, KAN, TTT, CHRASS, MCTS or
  OxtaMem;
- use faithful Mamba-2 mode (`mamba2_faithful=true`);
- make streaming incremental decoding the default;
- select GPU automatically when available and fall back to CPU;
- accept synthetic datasets from in-memory strings, JSON and JSONL.

## Import

After building or copying `nsos_ext`, add both paths to Python:

```python
import sys

# directory containing nsos_ext.so / nsos_ext.pyd
sys.path.insert(0, "/content/nsos_ext_mambavs")

# repository Python package path
sys.path.insert(0, "/content/reimagined/OXN/nsos/python")

from nsos_mamba import (
    CharTokenizer,
    MambaModuleConfig,
    NSOSMamba,
    TextPair,
    load_text_pairs_json,
    load_text_pairs_jsonl,
)
```

In the current benchmark notebook, `nsos_ext` is already imported from
`/content/nsos_ext_mambavs`; only the repository Python path may need to be
inserted.

## Device and streaming flags

Environment flags:

```bash
NSOS_MAMBA_DEVICE=auto      # auto | gpu | cuda | cpu
NSOS_MAMBA_STREAMING=1      # 1 default, 0 disables incremental streaming
NSOS_MAMBA_LAYERS=12
NSOS_MAMBA_DMODEL=128
NSOS_MAMBA_DSTATE=64
NSOS_MAMBA_CONTEXT=4096
```

Behavior:

- `auto` uses GPU when CUDA is visible through PyTorch;
- if GPU construction fails and `allow_cpu_fallback=True`, the wrapper rebuilds
  the model on CPU;
- `streaming=True` calls `model.set_streaming_inference(True)` and generation
  uses one prefill plus one-token incremental steps.

## Minimal synthetic example

```python
from nsos_mamba import CharTokenizer, MambaModuleConfig, NSOSMamba, TextPair

pairs = [
    TextPair("Quem é você?", "Oxta"),
    TextPair("quem é você?", "Oxta"),
    TextPair("Quem e voce?", "Oxta"),
]

tok = CharTokenizer.from_texts(
    [x.prompt for x in pairs] + [x.answer for x in pairs]
)

cfg = MambaModuleConfig(
    vocab_size=tok.vocab_size,
    num_layers=12,
    d_model=128,
    device="auto",
    streaming=True,
)

mamba = NSOSMamba(cfg)

def progress(step, loss):
    print(f"[train] step {step} loss~{loss:.4f}")

mamba.fit_text_pairs(
    pairs,
    tok,
    steps=1200,
    batch_size=16,
    learning_rate=3e-3,
    callback=progress,
)

result = mamba.generate_text("Quem é você?", tok, max_new_tokens=8)
print("resposta:", repr(result.text))
print("ids:", result.generated_ids)
print("decode tok/s:", result.decode_tokens_per_sec)
print("e2e tok/s:", result.end_to_end_tokens_per_sec)
```

Expected output after convergence:

```text
resposta: 'Oxta'
```

## Decode benchmark

The benchmark separates prefill from incremental decode:

- `prefill_ms`: time to process the prompt and produce the first next-token
  logits;
- `first_incremental_ms`: time of the first one-token decode after prefill;
- `decode_tokens_per_sec`: steady decode throughput after prefill;
- `end_to_end_tokens_per_sec`: generated tokens divided by prefill + decode.

```python
prompt_ids = tok.encode("Quem é você?", bos=True)
bench = mamba.benchmark_decode(prompt_ids, max_new_tokens=1024, repeats=5)
bench.print()
```

For edge reporting, prefer the steady streaming number and always include:

- model size (`num_layers`, `d_model`);
- vocabulary size;
- prompt length;
- generated token count;
- device/GPU model;
- whether streaming was enabled.

## Dataset formats

### In-memory strings

```python
from nsos_mamba import TextPair

pairs = [
    TextPair(prompt="Quem é você?", answer="Oxta"),
    TextPair(prompt="Qual seu nome?", answer="Oxta"),
]
```

### JSON

`dataset.json`:

```json
[
  {"prompt": "Quem é você?", "answer": "Oxta"},
  {"prompt": "Qual seu nome?", "answer": "Oxta"}
]
```

Load:

```python
from nsos_mamba import load_text_pairs_json

pairs = load_text_pairs_json("dataset.json")
```

Also accepted:

```json
{"data": [{"prompt": "...", "answer": "..."}]}
```

or:

```json
{"examples": [{"prompt": "...", "answer": "..."}]}
```

### JSONL

`dataset.jsonl`:

```jsonl
{"prompt": "Quem é você?", "answer": "Oxta"}
{"prompt": "Qual seu nome?", "answer": "Oxta"}
```

Load:

```python
from nsos_mamba import load_text_pairs_jsonl

pairs = load_text_pairs_jsonl("dataset.jsonl")
```

## Notes and limitations

- `CharTokenizer` is for demos/probes.  Real training should use the project
  tokenizer pack.
- The wrapper uses greedy decoding.  Sampling can be added later without
  changing the Mamba architecture.
- Streaming incremental is the default because edge latency depends on avoiding
  full-context recomputation per generated token.
- This module is a productization layer.  It should not be used to make new
  benchmark claims unless the benchmark also reports the official reference and
  exact model configuration.
