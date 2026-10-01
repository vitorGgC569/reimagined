# Notas internas pro Vitor — NÃO MANDAR PRO AMIGO

> Este doc fica fora do bundle. É só pra você saber o que está rodando lá.
> Se ele perguntar algo específico, você pode consultar isso.

## O que ele recebe

`OxtaTrainer_v1.7z` — 10.35 GB (11,116,408,266 bytes exatos).

Conteúdo (190 arquivos, 6 pastas):
- `OxtaTrainer.exe` (1.8 MB)
- `_internal/` (1.4 GB — Python 3.11 runtime + nsos_ext.pyd + 5 CUDA DLLs + ML libs)
- `data/` (9.9 GB)
  - `bundle/tokenizer_8192.ox3` (137 KB — v4 com 7933 BPE merges)
  - `config/runtime.json` (perfil 200M chinchilla25b)
  - `datasets/corpus_a_0000..0119.jsonl.zst` (FineWeb-2 PT, 120 shards)
  - `datasets/corpus_b.jsonl` (BR-TaxQA, 181 MB)
  - `datasets/corpus_d.jsonl` (LeNER-Br, 949 KB)
  - `manifest.json` (SHA256 de tudo)
- `README.txt` (instruções pro amigo)

## Configuração que vai rodar

- Arquitetura: 20 layers, d_model=1024, n_heads=16, n_kv_heads=4
- MoE: 8 experts top-2 (slot 3 de cada 3 blocos)
- Attention: sliding window 4096, slot 2 de cada 2 blocos
- BitNet 1.58-bit ternary + Slender embedding + bitmamba LUT
- Batch size: 12, ctx: 1024, gradient checkpointing ON
- Target: 50.000 steps (~6 dias na RTX 2080 Ti)
- Compute: sm_75 (Turing) — kernels nativos, sem JIT

## Caveats reais (o amigo NÃO precisa saber, você sim)

### 1. Learning rate efetivo é 0.001, não 5.5e-4 (configurado)
- `nsos_sdk.cpp:1047` cria `Trainer(model, 0.001f)` hardcoded
- O `runtime.json` declara `learning_rate: 0.00055` mas o engine ignora
- Impacto: LR ~1.8x mais alto que ideal. Treino converge com mais ruído.
  Gradient clipping (max_grad_norm=0.9) protege contra blow-up.
- Para corrigir no futuro: expor `set_learning_rate` no `InferenceEngine` binding.

### 2. Resume não preserva optimizer state
- `save_checkpoint(.bin)` salva só `model->save(path)` — pesos apenas
- `load_model(.bin)` cria `Trainer` novo do zero (Adam m,v = 0)
- Impacto por pausa: ~50-100 steps de "spike" enquanto Adam reestabiliza
- 5 pausas em 50k steps = ~0.5% de progresso desperdiçado. Aceitável.
- Para corrigir: serializar/desserializar Adam state junto com weights.

### 3. Datasets baixados
- FineWeb-2 PT (substitui CulturaX que ficou gated)
- ~120 shards × 256 MB = 30 GB raw, comprimido com zstd pra 9.7 GB
- BR-TaxQA: 8388 docs IRPF + CARF (gold para SFT)
- LeNER-Br via mirror `eduagarcia/portuguese_benchmark` (peluz/lener_br quebrou)
- BACEN: todos os 3 mirrors sumiram do HF. Não está no bundle (2 MB, não é blocker).

### 4. Tokenizer
- `tokenizer_8192.ox3` v4 (May 15 2026), não v11 do Drive
- 7933 BPE merges, vocab 8189 + 3 specials
- Compressão PT: ~1.77 chars/token (subótimo — não foi treinado primarily em PT)
- v11 do Drive seria melhor mas v4 funciona

### 5. GPU pre-flight check
- `trainer_main.py:check_gpu()` valida compute_cap via nvidia-smi
- Aborta limpo se < sm_75 (mensagem clara em PT)
- 2080 Ti = sm_75 = passa direto
- 30xx, 40xx via PTX JIT (testado conceitualmente, não na prática)

## Quando ele te mandar `output/final_pack/` de volta

Vai conter:
- `model.nsos.bin` (pesos quantizados, ~50-100 MB)
- `tokenizer.nsos`
- `config.nsos`
- `manifest.nsos`

Pra carregar local:
```python
import nsos_ext
engine = nsos_ext.InferenceEngine()
engine.load_model('caminho/para/final_pack/', nsos_ext.ModelConfig())
# Vai detectar o manifest.nsos e fazer try_load_model_pack
```

## Hashes pra integridade
- Tokenizer SHA256: `d52af8996967080acadf7efebff71b380d7068a8f2008e26531967dd1db82582`
- runtime.json: 966 bytes
- Archive total: 11,116,408,266 bytes
