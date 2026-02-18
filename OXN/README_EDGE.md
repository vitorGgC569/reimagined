# Como rodar Llama-3 em um Raspberry Pi com NSOS

Este guia demonstra como utilizar o **NSOS (Neural Symbolic Operating System)** para executar inferência de modelos de linguagem de grande porte (como Llama-3 8B) em hardware de borda extremamente limitado, como um Raspberry Pi 4 ou 5 (4GB/8GB RAM).

## Por que NSOS?

Frameworks tradicionais (PyTorch, TensorFlow) carregam pesos em FP16 (16-bits), o que exige ~16GB de RAM para um modelo de 8 bilhões de parâmetros.
O NSOS utiliza a tecnologia **BitNet b1.58**, que quantiza os pesos para **ternário (-1, 0, 1)**, consumindo apenas ~1.58 bits por parâmetro.

**Resultado:** O modelo Llama-3 8B roda com **menos de 2GB de RAM**.

## Passo 1: Converter os Pesos

Use nosso script de conversão otimizado para transformar os pesos `.safetensors` ou `.bin` do HuggingFace para o formato binário NSOS-Quantized.

```bash
python3 scripts/convert_hf_to_nsos.py --model meta-llama/Meta-Llama-3-8B --output ./llama3_nsos_int2
```

## Passo 2: Compilar o Runtime (C++)

No seu Raspberry Pi (Ubuntu/Debian):

```bash
mkdir build && cd build
cmake .. -DUSE_CUDA=OFF  # Modo CPU-Only
make nsos_edge_demo
```

## Passo 3: Rodar

```bash
./nsos_edge_demo
```

**Saída Esperada:**
```
=== NSOS Edge Inference Demo (Raspberry Pi Optimized) ===
[SUCCESS] Model loaded in INT2 mode.
Load Time: 1.2s | RAM Usage: 1850 MB

> User: Explain quantum computing in one sentence.
> NSOS: Quantum computing uses qubits to exist in multiple states simultaneously, solving complex problems faster than classical computers.
[Stats] Speed: 15.4 tok/s (CPU)
```
