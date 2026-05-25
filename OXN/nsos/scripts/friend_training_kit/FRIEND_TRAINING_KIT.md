# Oxta Contábil 200M — Training Kit pro Amigo

> Tudo que você precisa pra treinar o modelo Oxta Contábil 200M no seu PC com RTX 2080 Ti.
> Esse documento assume que você é o amigo do Vitor que aceitou rodar o treino.  Obrigado :)
>
> **Tempo total:** ~30-60 min de setup + ~6 dias de treino contínuo (ou ~9-10 dias se você usar o PC durante o dia).

---

## O que vai acontecer

Você vai treinar a primeira versão do modelo Oxta Contábil — um modelo de linguagem brasileiro 200M de parâmetros especializado em domínio fiscal/contábil, otimizado pra rodar em CPU usando quantização 1.58-bit (ternária).

O resultado final é **um arquivo de ~80MB** (`final_edge_linear.nsos`) que é o "cérebro" do modelo treinado.  Você manda esse arquivo de volta pro Vitor.

---

## Pré-requisitos no PC

| Item | Versão mínima | Como verificar |
|------|---------------|----------------|
| Windows 10/11 ou Linux Ubuntu 22.04+ | — | `winver` ou `uname -a` |
| NVIDIA Driver | 535+ | `nvidia-smi` |
| CUDA Toolkit | 12.5+ | `nvcc --version` |
| Visual Studio 2022 com **C++ workload** (Win) **OU** GCC 11+ (Linux) | — | `cl /?` ou `g++ --version` |
| CMake | 3.18+ | `cmake --version` |
| Python | 3.11+ | `python --version` |
| Git | qualquer recente | `git --version` |
| Espaço em disco | **80 GB livre** | (build + dataset + checkpoints) |
| RAM | 16 GB | (32 GB recomendado) |
| GPU | RTX 2080 Ti 11GB confirmada | `nvidia-smi` |

Se alguma coisa faltar, instala antes de prosseguir.  Links úteis:
- CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- CMake: https://cmake.org/download/
- Python: https://www.python.org/downloads/
- Visual Studio Community 2022 (Windows): https://visualstudio.microsoft.com/

---

## Passo 1 — Clone do repositório (5 min)

Você vai precisar de um Personal Access Token do GitHub que o Vitor te mandou (formato `github_pat_...`).

```powershell
# Substitui SEU_PAT pelo token que o Vitor te mandou
git clone https://SEU_PAT@github.com/vitorGgC569/reimagined.git
cd reimagined
git checkout oxta-contabil
```

Isso baixa ~50MB.

---

## Passo 2 — Bundle de tokenizer + curriculum (uma vez, 2 min)

O modelo precisa de um "bundle" com o tokenizer BPE pré-treinado e os dados de currículo.  Esse arquivo é grande (~50MB) e não vai no GitHub.

**O Vitor vai te mandar separado:** um ZIP chamado `distillation_bundle_v11.zip`.

Extrai dentro do repo na pasta certa:

```powershell
# Windows PowerShell
Expand-Archive distillation_bundle_v11.zip -DestinationPath OXN\nsos\scripts\
```

```bash
# Linux/WSL
unzip distillation_bundle_v11.zip -d OXN/nsos/scripts/
```

A pasta resultante deve ser `OXN/nsos/scripts/distillation_bundle_v11/` contendo `tokenizer_8192.ox3` entre outros.

---

## Passo 3 — Rodar o treino (1 comando)

Tudo é automatizado.  Só rode:

**Windows:**
```powershell
cd OXN\nsos\scripts\friend_training_kit
.\train_oxta_contabil_200m.bat --full
```

**Linux / WSL:**
```bash
cd OXN/nsos/scripts/friend_training_kit
chmod +x train_oxta_contabil_200m.sh
./train_oxta_contabil_200m.sh --full
```

A flag `--full` baixa dataset completo (~30GB).  Sem flag, baixa só 5GB que é suficiente pra começar mas qualidade menor.

### O que o script faz automaticamente:

1. **Verifica pré-requisitos** (CUDA, Python, GPU, etc.)
2. **Instala libs Python** (`pip install datasets huggingface_hub ...`)
3. **Build do NSOS C++** — primeira vez leva 15-25 min, depois é instantâneo
4. **Baixa datasets brasileiros** (CulturaX PT-BR, BR-TaxQA-R com acórdãos CARF, BACEN FAQ, LeNER-Br)
5. **Aplica power limit 220W** na GPU (reduz temperatura, perde só ~5% performance)
6. **Inicia o treino** com checkpoint automático cada 100 steps

A partir daqui você pode fechar o terminal — o treino continua em background.

Wait, mentira.  Se você fechar o terminal vai matar o processo.  Veja a seção "Rodar em background" abaixo se quiser fechar.

---

## Passo 4 — Acompanhar progresso

O script imprime no terminal a cada 25 steps algo como:

```
[train] phase3_curated_text: samples=30000 mixed=30000 ... steps=900 ...
  step=100 loss=8.7234 ema=8.7891 lr=1.50e-04
  step=200 loss=7.9821 ema=8.2345 lr=2.85e-04
  step=300 loss=7.3214 ema=7.5678 lr=4.20e-04
  ...
```

**Sinal de saúde:** loss desce gradualmente.  Começa em ~9.5 (random init) e deve cair pra **~3-4** ao fim do treino completo.  Variações pra cima dentro de cada fase são normais (mudança de distribuição entre dados).

Os checkpoints são salvos em `oxta_packs/rtx2080ti_200m_YYYYMMDD_HHMM/` — você verá vários arquivos `.bin` aparecerem durante o treino.  O último `final_model.bin` + `final_edge_linear.nsos` é o resultado final.

---

## Passo 5 — Após o treino terminar

Quando aparecer:
```
============================================================
 Treino COMPLETO
 Pack final:    .../final_edge_linear.nsos
============================================================
```

Você manda pro Vitor APENAS o arquivo `final_edge_linear.nsos` (~80MB).  Pode mandar via Drive, WeTransfer, ou compactar pra ZIP.

Os outros checkpoints intermediários (`phase1_algorithms.bin`, etc.) você pode apagar pra liberar espaço — ele só precisa do final.

---

## Rodar em background (opcional, mas recomendado pra 6 dias)

**Windows — usando Task Scheduler ou Start-Process:**
```powershell
Start-Process powershell -ArgumentList "-NoExit","-Command",".\train_oxta_contabil_200m.bat --full" -WindowStyle Hidden
```

**Linux — usando nohup ou tmux:**
```bash
# Opção 1: nohup (mais simples)
nohup ./train_oxta_contabil_200m.sh --full > train.log 2>&1 &
echo $! > train.pid    # salva PID pra matar depois

# Opção 2: tmux (melhor — pode reanexar)
tmux new -s oxta
./train_oxta_contabil_200m.sh --full
# Ctrl+B depois D pra desanexar (treino continua)
# Pra reanexar: tmux attach -t oxta
```

---

## Se algo der errado

### "CUDA out of memory"
- Significa que VRAM não bastou.  Provavelmente outro processo usando GPU.
- Solução: feche jogos, navegadores com GPU acceleration, OBS, etc.
- Verifica: `nvidia-smi` — se aparecer só "python", tá ok.

### "build failed" na etapa 3
- Provavelmente CUDA Toolkit ou Visual Studio mal instalado.
- Verifica: `nvcc --version` deve responder.  `cl /?` no PowerShell.
- Solução: reinstala CUDA Toolkit 12.5+ COM "Visual Studio Integration" marcado.

### "Bundle not found"
- Você esqueceu de extrair o `distillation_bundle_v11.zip` do passo 2.
- Confere: `dir OXN\nsos\scripts\distillation_bundle_v11\tokenizer_8192.ox3` deve existir.

### Travou / parou de imprimir
- Provavelmente power outage ou crash da GPU.
- Os checkpoints estão salvos.  Rode:
  ```powershell
  .\train_oxta_contabil_200m.bat --resume
  ```
  Vai retomar do último checkpoint salvo (perde no máximo ~25 min).

### GPU passando 85°C constante
- Risco de throttle.  Reduz power limit pra 200W:
  ```powershell
  nvidia-smi -pl 200
  ```
- Verifica ventoinhas, poeira no PC.

---

## O que esperar de tempo

| Estágio | Tempo |
|---------|-------|
| Setup (etapas 1-3) | 30-60 min |
| Download datasets `--full` | 30-90 min (depende da net) |
| Treino — fase 1 (algorithms) | ~3h |
| Treino — fase 2 (structured) | ~2h |
| Treino — **fase 3 (curated text)** ⭐ | ~80h (a maior, é onde aprende coerência) |
| Treino — fase 4 (instructions) | ~10h |
| Treino — fase 5 (verifier) | ~7h |
| Treino — fase 6 (memory) | ~10h |
| **Total treino** | **~140h = ~6 dias 24/7** |
| Eval + pack final | ~10 min |

Se você usar o PC pra trabalho 4-6h/dia, treina o restante (18-20h/dia), o tempo total vira **~9-10 dias calendar**.

---

## Custo de eletricidade aproximado

```
220W × 24h × 6 dias = 31.7 kWh
× R$ 0.80/kWh = R$ 25.40 total
```

R$ 25 pelo treino completo.  Vitor te ressarce (combina com ele).

---

## Por que isso importa

Você está treinando **o primeiro modelo de IA brasileiro 200M de domínio fiscal**, do zero, em casa.  Não é exagero — não existe outro com esse perfil específico.

A partir do `final_edge_linear.nsos` que você vai gerar, o Vitor monta a app desktop Oxta Contábil, que escritórios de contabilidade brasileiros vão usar **sem mandar dados de cliente pra nuvem americana**.  Conformidade LGPD por construção, soberania de dados real, e tudo começa com 6 dias da sua RTX 2080 Ti.

Obrigado por topar.  Qualquer dúvida durante o treino, chama o Vitor.

---

## Arquivos importantes

| Caminho | O que é |
|---------|---------|
| `OXN/nsos/scripts/friend_training_kit/train_oxta_contabil_200m.bat` | Launcher Windows |
| `OXN/nsos/scripts/friend_training_kit/train_oxta_contabil_200m.sh` | Launcher Linux |
| `OXN/nsos/scripts/friend_training_kit/FRIEND_TRAINING_KIT.md` | Este arquivo |
| `OXN/nsos/scripts/distillation_bundle_v11/` | Bundle (Vitor manda separado) |
| `OXN/nsos/build-rtx2080ti/` | Build C++ (gerado automaticamente) |
| `~/oxta_data/` ou `%USERPROFILE%\oxta_data\` | Datasets baixados |
| `~/oxta_packs/rtx2080ti_200m_*/` | Checkpoints + pack final |
