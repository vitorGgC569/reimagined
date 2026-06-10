# ENGRAMA — arquitetura memória-nativa ("assenta e lembra")

**Status:** pesquisa / incubação (não é claim de produto). Evolução e superset de
`LHR_PENDULO_DESIGN.md` após autocrítica. Default OFF atrás de flags.

---

## 0. Autocrítica do Pêndulo v1 (o que estava errado e como conserta)

1. **Física decorativa.** v1 tratou `F(z)=f_θ(z)−z` como gradiente de um
   potencial. Um bloco Transformer não é campo conservativo (Jacobiano não
   simétrico) ⇒ não há Lyapunov garantido; "convergência por construção" estava
   supervendida. v1 é, na verdade, heavy-ball fixed-point com parada por resíduo.
   **Conserto (v2):** construir o core de assentamento com componentes cuja
   descida de energia é PROVÁVEL — atenção-Hopfield moderna (Ramsauer et al.:
   atenção = passo de descida de `E(z) = −lse(β, Ξᵀz) + ½‖z‖²`) + MLP simétrico
   (W, Wᵀ), receita do Energy Transformer (Hoover et al.). Parada por ΔE vira
   rigorosa. Custo honesto: simetria restringe expressividade ⇒ apenas o CORE é
   energia-restrito; o tronco Jamba permanece padrão.
2. **"Iterar é grátis" é claim de INFERÊNCIA** (kernels ternários packed no
   edge), não de treino (QAT/BF16 paga custo cheio por iteração).
3. **Aposta não provada em 40M/20M tokens.** Extrapolação algorítmica por
   recorrência está estabelecida em escala pequena (Schwarzschild et al.);
   transferência para linguagem nessa escala é hipótese ⇒ o gate científico é
   algorítmico primeiro (curva de pensamento), linguagem depois.

## 1. O padrão do projeto que a v1 não nomeou

Componentes por função: KV/estado Mamba (ms) · HAM bind/bundle (contexto) ·
H-TTT (formação de memória em test-time) · CHRASS (memória relacional/topologia)
· MemorySystem causal + OxtaMem disco (persistente) · MCTS (deliberação).

⇒ O NSOS já é, de fato, um **modelo memória-nativo com cinco memórias em cinco
escalas de tempo**. ENGRAMA assume isso como princípio organizador.

## 2. A arquitetura

| Escala | Mecanismo | Papel |
|---|---|---|
| ms | KV/Mamba + HCB holográfico (gist O(1) do passado distante) | contexto |
| segundos | **Pêndulo v2**: core Hopfield-energia weight-tied, iterado até ΔE<ε | **LER** = assentar a energia sobre os slots |
| sessão | **H-TTT repontado**: atualiza SOMENTE os slots de memória | **ESCREVER** — bounded, interpretável, serializável |
| persistente | OxtaMem (disco) realimenta os slots | longo prazo |

**Acoplamento leitura/escrita (novo):** slots são UM tensor com duas dinâmicas.
Leitura = descida de energia até os padrões. Escrita = atualização simplética
dos padrões com força ∝ **energia residual no assentamento**:
*"escreva o que você falhou em recuperar"*. (Titans usa ‖grad‖ como surpresa;
energia-de-assentamento é mais barata e cai naturalmente do formalismo.)

### Botões cognitivos
γ (fricção) = orçamento de compute · τ (temperatura) = self-consistency latente
(k trajetórias do mesmo prefixo + voto) · r_max = teto de pensamento ·
ε = sensibilidade da parada · η_write × surpresa = taxa de escrita.

## 3. Por que ataca os três eixos

- **Inteligência / déficit Chinchilla (40M params vs ~20M tokens, ~40× abaixo):**
  params para HABILIDADE, conhecimento no STORAGE (argumento RETRO /
  Memorizing Transformers: memória externa compra ordens de magnitude de
  parâmetros). Treinar COM retrieval a 40M cabe numa T4.
- **Generalização:** recorrência weight-tied ⇒ algoritmos/extrapolação;
  retrieval ⇒ conhecimento sem decoreba nos pesos. Ambos atacam o gap
  "treino lindo, held-out fraco".
- **Produto edge:** `model.bin` compartilhado + **`user.engrama` (~256KB,
  BF16 [N×d] + deltas + metadados)** por usuário: o modelo lembra do usuário,
  no dispositivo, sem fine-tune, com replay determinístico (módulo determinism)
  e rollback pelo self-healer. Privacidade por construção. Combinação não
  existente no mercado.

## 4. Novidade honesta vs estado da arte

Já existem: escrita test-time (Titans, TTT-linear/Sun), leitura por energia
(Hopfield moderno, Energy Transformer), recorrência latente (Geiping), retrieval
(RETRO). **Inédita é a síntese edge:** leitura-por-energia COM parada física +
escrita bounded/serializável por usuário + hierarquia até disco + corpo ternário
+ determinismo/self-healing industrial. Flag-plant de benchmark:
**accuracy-per-joule de raciocínio no edge** (iteração ternária ADD/SUB tende a
vencer densos nessa métrica; ninguém mede raciocínio assim).

Autocrítica das irmãs: **HCB** = capacidade VSA ~O(d/log d) (≈50-100 itens a
d=512) ⇒ memória de *gist*, não verbatim; valor sobre o estado do Mamba é o
endereçamento por conteúdo — provar. **TTT²/CHRASS** = o repo mantém
use_chrass=false por falta de validação ⇒ prioridade baixa até topologia
injetada provar valor.

## 5. Roadmap (absorve Fases 3-5 antigas)

- **M0 — semana de generalização (T4):** harness de eval honesto (PPL held-out +
  sondas algorítmicas: adição multi-dígito, paridade, parênteses) + **Fused
  Linear-CE estilo Liger** (não materializar logits [B·T,V], ~100MB/step) +
  números-base. Sem isso nada é mensurável.
- **M1 — Pêndulo v2:** core Hopfield-energia (flag `use_latent_reasoner`, OFF).
  Gates: paridade-OFF byte-idêntica · determinismo (τ=0) · **curva de
  pensamento** (acurácia ↑ monotônica com r; r_test>r_train ⇒ extrapolação).
- **M2 — Memória de sessão:** banco de slots + escrita H-TTT gated por surpresa
  + `user.engrama` save/load + rollback self-healer. Gates: fato ensinado na
  sessão persiste após restart · zero regressão com banco vazio.
- **M3 — Conhecimento:** treino com retrieval (OxtaMem→slots) + HCB.
  Gate: held-out com/sem memória (o delta É a tese RETRO).
- **M4 — System 2 + flag-plant:** MCTS guiado por energia · benchmark
  accuracy-per-joule publicável.

## 6. Sinergia com a infra desta branch

Pool de memória (a0618ed) ⇒ endereços estáveis ⇒ **CUDA Graphs** no core
weight-tied (capturar 1 iteração, replay r). BF16 runtime (1231699) p/ treino.
Profiler NVML + NSOS_TRAIN_TIMING (348e512) já provaram que fwd+bwd dominam ⇒
Fused-CE é o próximo lever de treino. KAN (medido mais rápido/leve que FFN no
GPU path) é candidato a componente do core de energia.
