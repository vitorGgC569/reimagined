# Roadmap Industrial: Projeto Marco Zero (AGI Científica)

Este documento detalha a transformação do protótipo experimental "Mad Science" em uma pipeline industrial robusta para treinar a AGI Científica.

---

## 🏭 1. O Pipeline Industrial (A Fábrica)

O objetivo é orquestrar o fluxo de dados massivo (Petabytes) para o Kernel 1.58-bit sem gargalos de CPU/Python.

### A Stack de Ferramentas (Minimalista & Rust-Aligned)
Rejeitamos "bloatware" corporativo em favor de ferramentas focadas em performance e integração C++.

*   **Configuração:** **Hydra** (Python). Permite composição dinâmica de configs (`conf/model/bitnet_158.yaml`, `conf/physics/hamiltonian.yaml`).
*   **Rastreamento:** **WandB (Weights & Biases)** (ou MLFlow leve). O WandB é superior para visualizar a "física" (energia, temperatura) em tempo real via gráficos customizados.
*   **Orquestração:** **Ray**. Para distribuir o carregamento de dados e o "Teacher" (Llama-3) em múltiplos nós/GPUs, enquanto o "Student" (OXTA) roda no nó principal.
*   **Build System:** **CMake + Ninja**. Essencial para compilação incremental rápida do Kernel C++.

### O Elo Perdido do I/O (SmartLoader Integration)
Atualmente, o `trainer.cpp` usa `std::ifstream` (lento) e o `benchmarks/full_scale_train.py` usa `DataLoader` do PyTorch (Python overhead).

**Solução: `AionDataLoader` (C++ -> Python Zero-Copy)**

Precisamos criar uma classe `AionDataLoader` em `OXB` que exponha o `SmartLoader` para o Python via `PyArrow` ou `DLPack`.

**Fluxo de Dados:**
1.  **Disco (.ox3):** Blocos binários comprimidos com LZ4.
2.  **SmartLoader (C++):** Thread `io_uring` lê blocos para um Ring Buffer em RAM (Pinned Memory).
3.  **AionDataLoader (Python Binding):**
    *   Não retorna `list` ou `tensor`.
    *   Retorna um **Handle (Ponteiro)** para o buffer da GPU/CPU.
4.  **OXN Kernel:** O `JambaModel::forward` aceita esse Handle e lê diretamente, sem cópia.

### Checkpointing Robusto (Architecture-Agnostic)
Para evitar incompatibilidade futura (ex: rollback da Química), o formato de checkpoint deve ser desacoplado.

*   **Formato:** **Safetensors** (padrão industrial) com metadados JSON no header.
*   **Estrutura:**
    ```
    checkpoints/v1.2/
    ├── model.safetensors       # Pesos 1.58-bit (int8 packing)
    ├── optimizer.safetensors   # Estados do AdamW/Muon (bf16)
    ├── topology.ox3            # Matriz ChrassLayer (Grafo)
    └── physics_state.json      # T, Friction, Momentum (H-TTT)
    ```
*   **Estratégia:** "Atomic Swap". Grave em `ckpt.tmp`, faça `fsync`, renomeie para `ckpt.last`.

---

## 📜 2. O Script Mestre de Treinamento (`train_master.py`)

Este script substitui os benchmarks isolados. Ele é o maestro.

```python
import hydra
import torch
from omegaconf import DictConfig
import aion_core  # SmartLoader
import nsos_ext   # O Kernel C++ (Student)
from transformers import AutoModelForCausalLM # O Teacher (Llama)

@hydra.main(config_path="conf", config_name="config")
def train_master(cfg: DictConfig):
    # 1. Setup da Fábrica
    logger = setup_wandb(cfg)
    dataloader = aion_core.SmartLoader(
        cfg.data.path,
        batch_size=cfg.train.batch_size,
        mode="streaming"
    )

    # 2. Setup dos Modelos
    # Student: O nosso BitNet 1.58b (C++)
    student = nsos_ext.JambaModel(cfg.model)
    if cfg.train.resume:
        student.load(cfg.train.checkpoint_path)

    # Teacher: Llama-3-8B (PyTorch/GPU) - Congelado
    teacher = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b", device_map="auto")
    teacher.eval()

    # 3. Otimizadores
    # Muon para a 'Mente' (Pesos), AdamW para 'Física' (Parâmetros H-TTT)
    opt_weights = nsos_ext.MuonOptimizer(student.parameters(), lr=cfg.train.lr)

    print("=== Iniciando Ciclo Industrial ===")

    step = 0
    temperature = cfg.physics.initial_temp

    while step < cfg.train.max_steps:
        # A. Ingestão Zero-Copy
        batch_handle = dataloader.next_batch() # Ponteiro C++

        # B. Warm-up (Primeiros 1k passos: Sem Física, Sem Teacher)
        if step < 1000:
            loss = student.forward_loss_supervised(batch_handle) # Next Token Prediction puro

        else:
            # C. Ciclo H-TTT (Modo Gênio)
            # A cada 10 passos, ativamos a simulação física completa
            use_physics = (step % 10 == 0)
            student.set_hamiltonian_mode(use_physics)

            # D. Forward do Aluno (Gera Logits 1.58b)
            student_logits = student.forward(batch_handle)

            # E. Forward do Professor (Gera Logits FP16 Rico)
            # Precisamos converter o batch_handle para Torch só para o Teacher
            with torch.no_grad():
                teacher_batch = batch_handle.to_torch()
                teacher_logits = teacher(teacher_batch).logits

            # F. Perda de Destilação (KL Divergence)
            # O aluno tenta imitar a distribuição de probabilidade do professor
            loss = distillation_loss(student_logits, teacher_logits, T=2.0)

            # G. Perda de Energia (Regularização)
            if use_physics:
                energy = student.get_kinetic_energy()
                loss += 0.01 * energy # Penaliza pensamento excessivo para coisas simples

        # H. Backward & Step
        student.backward(loss)
        opt_weights.step()

        # I. Eval Loop & Sanity Check
        if step % 500 == 0:
            sanity_score = run_logic_probe(student) # Teste simples (2+2)
            logger.log({"loss": loss, "sanity": sanity_score, "temp": temperature})
            save_checkpoint(student, step)

        step += 1

```

---

## 🎓 3. O Currículo Científico (A Educação)

Não podemos jogar a Wikipédia inteira de uma vez. O cérebro ternário satura.

### Fase 1: Alfabetização (A Base)
*   **Objetivo:** Dominar sintaxe, gramática e estrutura de frases.
*   **Dados:**
    *   **CulturaX (PT/EN):** Filtrado para alta qualidade (excluir spam/SEO).
    *   **Livros (Gutenberg):** Narrativas longas para treinar a "Janela de Atenção".
*   **Estratégia:** Treino Supervisionado Puro (Next Token Prediction). H-TTT desligado (System 1 apenas).
*   **Meta:** Perplexity < 10 no conjunto de validação WikiText.

### Fase 2: O Trivium Lógico (Aprender a Pensar)
*   **Objetivo:** Calibrar o H-TTT. O modelo deve aprender que "pensar custa energia, mas resolve problemas".
*   **Dados:**
    *   **OpenWebMath / AMPS:** Problemas de matemática passo-a-passo.
    *   **Synth-Logic:** Gerar 10M de exemplos de silogismos (`Se A>B e B>C, então A>C`) e códigos Python simples.
*   **Estratégia:** Ativar H-TTT. Usar **Reinforcement Learning (GRPO)** ou Destilação de CoT (Chain of Thought).
    *   Se o modelo errar a lógica, aumentamos a temperatura na próxima tentativa.

### Fase 3: A Especialização (Cientista)
*   **Objetivo:** Ler o mundo físico.
*   **Dados de Química:**
    *   **SMILES Strings:** Trate moléculas como texto (`CC(=O)OC1=CC=CC=C1C(=O)O` é Aspirina).
    *   **Topologia:** Injete o grafo molecular na `ChrassLayer` durante o treino. O modelo "sente" a molécula.
*   **Dados de Física:**
    *   **ArXiv (Physics):** LaTeX tokenizado.
    *   **Simulações:** Dados numéricos de trajetórias (pêndulos, órbitas).
*   **Prevenção de Catastrophic Forgetting:**
    *   **Replay Buffer:** Mantenha 10% dos dados do batch vindo da Fase 1 (Português) durante o treino de Química.
    *   **EWC (Elastic Weight Consolidation):** Congele os pesos vitais da gramática (identificados via Fisher Information matrix) para que a Química ocupe apenas os neurônios livres.

---

**Próximos Passos:**
1.  Implementar `AionDataLoader` (C++).
2.  Configurar `Hydra` + `WandB`.
3.  Rodar a **Fase 1** (Alfabetização) em uma fatia pequena (10GB) para validar o Kernel.
