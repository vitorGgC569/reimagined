# NSOS Mamba

`nsos-mamba` é um wrapper Python simples para usar apenas o módulo Mamba do
NSOS. Ele configura automaticamente o modelo em modo Mamba-2 faithful, ativa
streaming incremental por padrão e oferece utilitários pequenos para treinar
pares `prompt -> resposta`.

Este pacote não instala o runtime nativo `nsos_ext`. Antes de usar, compile ou
copie o `nsos_ext.so`/`nsos_ext.pyd` e deixe-o acessível via `PYTHONPATH` ou
pela variável `NSOS_EXT_PATH`.

## Instalação

```bash
pip install nsos-mamba
```

Se o `nsos_ext` estiver em uma pasta separada:

```bash
export NSOS_EXT_PATH=/caminho/para/nsos_ext
```

No Colab:

```python
import sys, os

sys.path.insert(0, "/content/nsos_ext_mambavs")
# ou:
os.environ["NSOS_EXT_PATH"] = "/content/nsos_ext_mambavs"
```

## Configurações principais

```python
from nsos_mamba import MambaModuleConfig

cfg = MambaModuleConfig(
    vocab_size=128,
    num_layers=12,
    d_model=128,
    device="auto",
    streaming=True,
)
```

Campos mais usados:

- `vocab_size`: tamanho do vocabulário.
- `num_layers`: número de camadas Mamba.
- `d_model`: largura do modelo.
- `d_state`: tamanho do estado SSM. Padrão: `64`.
- `max_context_tokens`: limite de contexto. Padrão: `4096`.
- `device`: `"auto"`, `"gpu"`, `"cuda"` ou `"cpu"`.
- `streaming`: ativa decode incremental. Padrão: `True`.
- `allow_cpu_fallback`: se a GPU falhar, cai para CPU. Padrão: `True`.

Variáveis de ambiente equivalentes:

```bash
NSOS_MAMBA_DEVICE=auto
NSOS_MAMBA_STREAMING=1
NSOS_MAMBA_LAYERS=12
NSOS_MAMBA_DMODEL=128
NSOS_MAMBA_DSTATE=64
NSOS_MAMBA_CONTEXT=4096
```

## Exemplo mínimo: "Quem é você?" -> "Oxta"

```python
from nsos_mamba import CharTokenizer, MambaModuleConfig, NSOSMamba, TextPair

pairs = [
    TextPair("Quem é você?", "Oxta"),
    TextPair("quem é você?", "Oxta"),
    TextPair("Quem e voce?", "Oxta"),
]

tok = CharTokenizer.from_texts(
    [p.prompt for p in pairs] + [p.answer for p in pairs]
)

cfg = MambaModuleConfig(
    vocab_size=tok.vocab_size,
    num_layers=12,
    d_model=128,
    device="auto",
    streaming=True,
)

mamba = NSOSMamba(cfg)

mamba.fit_text_pairs(
    pairs,
    tok,
    steps=1200,
    batch_size=16,
    learning_rate=3e-3,
    callback=lambda step, loss: print(f"[train] step {step} loss~{loss:.4f}"),
)

result = mamba.generate_text("Quem é você?", tok, max_new_tokens=8)

print("resposta:", repr(result.text))
print("tokens/s:", result.decode_tokens_per_sec)
```

Saída esperada:

```text
resposta: 'Oxta'
```

## Streaming incremental

O streaming incremental é o padrão. O fluxo usado é:

1. processa o prompt uma vez (`prefill`);
2. gera cada token novo passando apenas o último token;
3. reaproveita o estado interno do modelo.

Benchmark:

```python
prompt_ids = tok.encode("Quem é você?", bos=True)
bench = mamba.benchmark_decode(prompt_ids, max_new_tokens=1024, repeats=5)
bench.print()
```

Métricas:

- `prefill_ms`: tempo para processar o prompt.
- `first_incremental_ms`: primeiro passo incremental após o prefill.
- `decode_tokens_per_sec`: tokens/s do decode incremental.
- `end_to_end_tokens_per_sec`: tokens/s incluindo prefill.

## Datasets

### Lista em memória

```python
from nsos_mamba import TextPair

pairs = [
    TextPair(prompt="Quem é você?", answer="Oxta"),
    TextPair(prompt="Qual seu nome?", answer="Oxta"),
]
```

### JSON

Arquivo `dataset.json`:

```json
[
  {"prompt": "Quem é você?", "answer": "Oxta"},
  {"prompt": "Qual seu nome?", "answer": "Oxta"}
]
```

Uso:

```python
from nsos_mamba import load_text_pairs_json

pairs = load_text_pairs_json("dataset.json")
```

Também aceita:

```json
{"data": [{"prompt": "...", "answer": "..."}]}
```

ou:

```json
{"examples": [{"prompt": "...", "answer": "..."}]}
```

### JSONL

Arquivo `dataset.jsonl`:

```jsonl
{"prompt": "Quem é você?", "answer": "Oxta"}
{"prompt": "Qual seu nome?", "answer": "Oxta"}
```

Uso:

```python
from nsos_mamba import load_text_pairs_jsonl

pairs = load_text_pairs_jsonl("dataset.jsonl")
```

## Observações

- `CharTokenizer` é para testes, demos e dados sintéticos pequenos.
- Para treino real, use um tokenizer próprio do projeto.
- O pacote expõe o Mamba isolado; não ativa Attention, MoE, KAN, TTT, CHRASS,
  MCTS ou memória externa.
- O pacote é uma camada de uso sobre `nsos_ext`, não uma reimplementação em
  Python.
