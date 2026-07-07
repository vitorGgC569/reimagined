# NSOS Mamba

`nsos-mamba` é um wrapper Python para usar somente o runtime neural
Mamba/Attention do NSOS. Ele já configura:

- Mamba-2 faithful;
- streaming incremental por padrão;
- GPU automática quando disponível;
- fallback para CPU;
- treino supervisionado simples `prompt -> resposta`;
- datasets em memória, JSON e JSONL;
- modo ternário/QAT opcional.

## Instalação

```bash
pip install nsos-mamba
```

### Runtime nativo

O pacote precisa do runtime nativo `nsos_ext`.

Na versão Windows/Python 3.11 publicada como wheel binária, o pacote já pode
vir com runtimes embutidos:

- `nsos_ext_cuda...pyd`, quando CUDA está disponível;
- `nsos_ext_cpu...pyd`, como fallback CPU.

Se você tiver um `nsos_ext` próprio, aponte para ele assim:

```bash
set NSOS_EXT_PATH=C:\caminho\para\nsos_ext
```

ou, no Python:

```python
import os
os.environ["NSOS_EXT_PATH"] = r"C:\caminho\para\nsos_ext"
```

No Colab:

```python
import os
os.environ["NSOS_EXT_PATH"] = "/content/nsos_ext_mambavs"
```

## Configuração do modelo

```python
from nsos_mamba import MambaModuleConfig, NSOSMamba

cfg = MambaModuleConfig(
    vocab_size=128,
    num_layers=12,
    d_model=128,
    device="auto",
    streaming=True,
)

model = NSOSMamba(cfg)
```

Campos principais:

- `vocab_size`: tamanho do vocabulário.
- `num_layers`: número de camadas.
- `d_model`: largura do modelo.
- `d_state`: tamanho do estado SSM. Padrão: `64`.
- `max_context_tokens`: contexto máximo. Padrão: `4096`.
- `device`: `"auto"`, `"gpu"`, `"cuda"` ou `"cpu"`.
- `streaming`: ativa decode incremental. Padrão: `True`.
- `allow_cpu_fallback`: se GPU falhar, cai para CPU. Padrão: `True`.

Variáveis de ambiente:

```bash
NSOS_MAMBA_DEVICE=auto
NSOS_MAMBA_STREAMING=1
NSOS_MAMBA_LAYERS=12
NSOS_MAMBA_DMODEL=128
NSOS_MAMBA_DSTATE=64
NSOS_MAMBA_CONTEXT=4096
```

## Mamba puro ou Mamba + Attention

Por padrão, o pacote expõe Mamba puro:

```python
cfg = MambaModuleConfig(vocab_size=128, use_attention=False)
```

Para ativar Attention junto com Mamba:

```python
cfg = MambaModuleConfig(
    vocab_size=128,
    use_attention=True,
    attention_period=2,
    attention_slot=1,
)
```

Interpretação:

- `use_attention=False`: só Mamba.
- `use_attention=True`: Mamba + Attention.
- `attention_period`: frequência das camadas de atenção.
- `attention_slot`: posição da atenção dentro do período.

O wrapper não ativa MoE, KAN, TTT, CHRASS, MCTS nem memória externa.

## Modo ternário/QAT

Para treinar usando QAT ternário:

```python
cfg = MambaModuleConfig(
    vocab_size=128,
    ternary=True,
    ternary_warmup_steps=100,
    ternary_start_step=300,
    ternary_regularization=1e-3,
)
```

Ou por chamada de treino:

```python
model.fit_text_pairs(pairs, tokenizer, qat=True)
```

Campos:

- `ternary`: liga o QAT ternário.
- `ternary_warmup_steps`: passos iniciais antes do regularizador.
- `ternary_start_step`: passo em que o forward passa a usar fake-quant.
- `ternary_regularization`: força do regularizador para pesos ternários.

Observação prática: QAT ternário normalmente não acelera o treino. Ele adiciona
fake-quant e STE. O benefício esperado é memória/deploy/quantização, não
necessariamente steps/s durante treinamento.

## Exemplo mínimo: "Quem é você?" -> "Oxta"

```python
from nsos_mamba import CharTokenizer, MambaModuleConfig, NSOSMamba, TextPair

pairs = [
    TextPair("Quem é você?", "Oxta"),
    TextPair("quem é você?", "Oxta"),
    TextPair("Quem e voce?", "Oxta"),
]

tokenizer = CharTokenizer.from_texts(
    [p.prompt for p in pairs] + [p.answer for p in pairs]
)

cfg = MambaModuleConfig(
    vocab_size=tokenizer.vocab_size,
    num_layers=12,
    d_model=128,
    device="auto",
    streaming=True,
)

model = NSOSMamba(cfg)

model.fit_text_pairs(
    pairs,
    tokenizer,
    steps=1200,
    batch_size=16,
    learning_rate=3e-3,
    callback=lambda step, loss: print(f"[train] step {step} loss~{loss:.4f}"),
)

result = model.generate_text("Quem é você?", tokenizer, max_new_tokens=8)

print("resposta:", repr(result.text))
print("tokens/s:", result.decode_tokens_per_sec)
```

Saída esperada:

```text
resposta: 'Oxta'
```

## Exemplo de comandos/redirect

Dataset em memória:

```python
pairs = [
    TextPair("aic", "/redirect aic"),
    TextPair("painel", "/redirect dashboard"),
    TextPair("login", "/redirect auth"),
    TextPair("ajuda", "/redirect help"),
]
```

Uso:

```python
tokenizer = CharTokenizer.from_texts(
    [p.prompt for p in pairs] + [p.answer for p in pairs]
)

model = NSOSMamba(MambaModuleConfig(vocab_size=tokenizer.vocab_size))
model.fit_text_pairs(pairs, tokenizer, steps=1000, batch_size=16)

print(model.generate_text("aic", tokenizer, max_new_tokens=32).text)
# /redirect aic
```

## Streaming incremental e velocidade

Streaming é o padrão. O fluxo é:

1. processa o prompt uma vez (`prefill`);
2. gera os próximos tokens passando só o último token;
3. reaproveita o estado interno do modelo.

Benchmark:

```python
prompt_ids = tokenizer.encode("Quem é você?", bos=True)
bench = model.benchmark_decode(prompt_ids, max_new_tokens=1024, repeats=5)
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

`dataset.jsonl`:

```jsonl
{"prompt": "Quem é você?", "answer": "Oxta"}
{"prompt": "Qual seu nome?", "answer": "Oxta"}
```

```python
from nsos_mamba import load_text_pairs_jsonl
pairs = load_text_pairs_jsonl("dataset.jsonl")
```

## Observações

- `CharTokenizer` serve para testes, demos e dados sintéticos pequenos.
- Para treino real, use um tokenizer próprio.
- O pacote é uma camada de uso sobre `nsos_ext`; não é uma reimplementação em
  Python.
