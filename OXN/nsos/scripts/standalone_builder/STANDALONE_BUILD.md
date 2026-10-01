# Standalone Bundle — Guia de Build (interno)

> **Este documento é pra você, Vitor.** Não vai pro bundle.
>
> Ele explica como gerar o arquivo `OxtaTrainer_v1.7z` (~15-20GB) que você
> envia pro amigo. Tempo estimado de execução: **~3-4 horas**, dominado
> pelo download de datasets (~1h) e PyInstaller (~10min).

---

## Pré-requisitos no SEU PC

Antes de começar, garanta que está tudo instalado:

| Item | Versão | Como verificar |
|------|--------|----------------|
| Windows 10/11 64-bit | — | `winver` |
| Visual Studio 2022 Build Tools | + "Desktop development with C++" workload | `where cl` |
| CUDA Toolkit | 12.5+ | `nvcc --version` |
| `CUDA_PATH` env var | aponta pro CUDA root | `echo %CUDA_PATH%` |
| CMake | 3.18+ | `cmake --version` |
| Python | 3.11+ | `python --version` |
| 7-Zip | qualquer recente | `where 7z` |
| Espaço em disco livre | **~80 GB** | (build + dataset + bundle) |

Se faltar algo, instale antes de continuar. Sem isso o script aborta.

---

## Pré-requisito IMPORTANTE: bundle de tokenizer

O script assume que existe `OXN/nsos/scripts/distillation_bundle_v11/`
no repo. Isso vem de **uma execução prévia do `train_v11.ipynb` no Colab**
que cria o bundle e cacheia no Drive.

Se você não tem esse bundle:
1. Abre `colab/train_v11.ipynb` no Colab
2. Roda a Cell 9 (data preparation) até criar o bundle
3. Baixa `distillation_bundle_v11.zip` do Drive
4. Extrai em `OXN/nsos/scripts/`

Sem o bundle, `prepare_data.py` avisa (mas continua). O .exe gerado
vai dar erro em runtime ("tokenizer not found"). Resolve antes.

---

## Comando único (50-minute happy path se tudo der certo)

Abre **x64 Native Tools Command Prompt for VS 2022** (importante — é onde
`cl.exe` está no PATH) e:

```cmd
cd OXN\nsos\scripts\standalone_builder
build_standalone.bat
```

Por default produz CulturaX de 15GB descomprimido. Pra bundle maior ou
comprimido:

```cmd
build_standalone.bat --full              REM CulturaX 30GB descomprimido (~28GB final)
build_standalone.bat --compress          REM CulturaX 15GB comprimido (~9GB final)
build_standalone.bat --full --compress   REM 30GB comprimido (~17GB final)  ← RECOMENDADO
```

**Recomendação:** `--full --compress`. Você ganha 30GB de dataset (dobra o
volume de treino), comprimido pra ~17GB que ainda cabe em WeTransfer Pro
(20GB) e Drive normal.

---

## O que cada etapa faz

| # | Etapa | Tempo aprox |
|---|-------|-------------|
| 1 | Verifica pré-requisitos | 1 s |
| 2 | Build `nsos_ext.pyd` via CMake (sm_75) | 15-25 min |
| 3 | Staging directory setup | < 1 s |
| 4 | **`prepare_data.py` — download + bake datasets** | **~1h** |
| 5 | PyInstaller (`pyinstaller_spec.spec`) | 8-12 min |
| 6 | Monta `OxtaTrainer_v1/` final | 5-15 min (cópia de ~17GB) |
| 7 | Compacta com 7z para `OxtaTrainer_v1.7z` | 20-40 min |

Total realista: **3-4 horas** com `--full --compress`, dominado pelo
download (depende muito da sua conexão).

---

## Resultado final

Após o sucesso, você tem dois artefatos:

```
OXN/nsos/scripts/standalone_builder/
├── OxtaTrainer_v1/             ← pasta com tudo descompactado
│   ├── OxtaTrainer.exe         (~70 MB)
│   ├── _internal/              ← Python + nsos_ext.pyd + CUDA DLLs
│   ├── data/                   ← ~17GB de dataset + bundle + config
│   └── README.txt              ← instruções pro amigo
└── OxtaTrainer_v1.7z           ← ~15-17GB pra enviar
```

---

## Antes de enviar — checklist

1. **Testa no seu PC primeiro.** Roda `OxtaTrainer_v1\OxtaTrainer.exe`
   e deixa rodar 5 minutos. Confere:
   - Janela abre sem erro
   - GPU é detectada (`nvidia-smi` aparece nos logs)
   - Loss aparece em `logs/session.log`
   - Pelo menos 1 checkpoint aparece em `output/checkpoints/`
   - Pode parar com Ctrl+C

2. **Apaga `output/` antes de enviar.** Senão você manda 1-2GB de
   checkpoints intermediários que não servem pra ele.

3. **Confere que nenhum texto vazador escapou.** Abre o README.txt e
   procura por: "Oxta", "Slender", "Nemotron", "Mamba", "BitNet",
   "Contábil". Se aparecer, edita.

4. **Sobe pra WeTransfer Pro / Drive.** Vai te dar um link único.
   Manda o link pro amigo via canal privado (não público).

---

## Como o amigo usa

Ele recebe um link. Baixa o `OxtaTrainer_v1.7z` (~15-17GB, ~1h de download
dependendo da net dele). Descompacta com 7-Zip ou WinRAR. Vê uma pasta
com `OxtaTrainer.exe` + `data/`. Lê o `README.txt` (1 página).

Dá duplo clique. Vai rodar 6 dias. No fim, manda de volta o
`output/final_pack/` (~100MB) via Drive ou WeTransfer.

---

## Limites honestos desta abordagem

### O que ESTÁ protegido
- ✅ Código C++ do runtime (nsos_ext.pyd é binário, requer eng reversa séria)
- ✅ Estratégia de treino (curriculum, profile, hyperparams escondidos em .pyc)
- ✅ Nomes do projeto (Oxta, Slender, Nemotron não aparecem no bundle)
- ✅ Dataset proprietário (filtragem + extração não óbvia)

### O que NÃO está totalmente protegido
- ⚠️ `runtime.json` é JSON legível. Mostra hyperparams (layers=20, d_model=1024).
- ⚠️ `trainer_main.pyc` pode ser decompilado parcialmente com `pyinstxtractor` + `decompyle3`. Estrutura geral fica visível, mas comentários e nomes de variáveis privadas não.
- ⚠️ Nome `nsos_ext.pyd` se ele inspecionar `_internal/`. Pode ser renomeado pra `core.pyd` num refinamento futuro (precisa editar o C++ build).
- ⚠️ Datasets dentro de `data/datasets/` são JSONL legível (texto em português). Ele pode ler o conteúdo. Pra esconder isso, criptografar com chave embutida no .exe (próximo nível de proteção).

Pra fins práticos com um amigo de confiança, esse nível de obscuridade é
mais do que suficiente. O verdadeiro moat é o modelo treinado final, não
os scripts de treino.

---

## Troubleshooting

### `build_standalone.bat` aborta em "cl.exe não encontrado"
Você não está em "x64 Native Tools Command Prompt for VS 2022".
Abre o atalho certo no Menu Iniciar e tenta de novo.

### CMake erro "CUDA architectures not specified"
Passa `-DNSOS_CUDA_ARCHITECTURES=75` (já está no script). Se ainda
der erro, atualiza CMake pra ≥ 3.24.

### PyInstaller "Failed to extract pyz"
Algum antivírus pegou o .exe. Adiciona exceção pro diretório de build
no Defender/AVG/Kaspersky e tenta de novo.

### `prepare_data.py` exit code != 0
Provavelmente download HF falhou. Se foi só CulturaX que falhou, o
bundle ainda vai ter BR-TaxQA + BACEN + LeNER (suficiente pra treinar,
só com menos dados). Se quer CulturaX, configura `HF_TOKEN` env var
e tenta de novo:

```cmd
set HF_TOKEN=hf_seu_token_aqui
build_standalone.bat --full --compress
```

### 7-Zip não está instalado
Baixa em https://www.7-zip.org/ e adiciona ao PATH. Ou pula etapa 7
e compacta a pasta `OxtaTrainer_v1/` manualmente.

### O .exe gerado falha no PC do amigo com "DLL missing"
Provavelmente uma DLL CUDA não foi bundleada. Lista os DLLs CUDA em
`OxtaTrainer_v1/_internal/` e compara com o erro. Adiciona o padrão
faltante em `pyinstaller_spec.spec` (variável `CUDA_DLL_PATTERNS`).

---

## Iteração rápida (durante desenvolvimento)

Se você só quer testar mudanças no Python sem rebuildar tudo:

```cmd
REM Modo dev: roda direto sem PyInstaller
cd OXN\nsos\scripts\standalone_builder
set PYTHONPATH=%PYTHONPATH%;..\..\build-rtx2080ti\Release
python trainer_main.py
```

Isso usa o `nsos_ext.pyd` que já está buildado, lê `data/` da pasta
atual, salva `output/` aí mesmo. Útil pra validar mudanças em
`trainer_main.py` ou `runtime.json` antes de re-bundle.
