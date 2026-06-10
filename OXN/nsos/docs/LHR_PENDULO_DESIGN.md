# LHR — Latent Hamiltonian Reasoning ("Pêndulo Cognitivo")

**Status:** pesquisa / incubação (não é claim de produto). Default OFF atrás de flag.
**Origem:** síntese dos componentes já existentes no NSOS — H-TTT (dinâmica
simplética), recurrent-depth latent reasoning (Geiping et al., arXiv 2502.05171)
e o corpo ternário 1.58-bit — numa mecânica que nenhum dos três campos usa.

---

## 1. Tese

"Pensar" = evoluir um sistema Hamiltoniano **dissipativo** no espaço latente.
Um grupo de blocos Jamba com pesos amarrados (weight-tied) é iterado como um
integrador **leapfrog com fricção**, e o modelo **para de pensar quando a
energia estabiliza** — um critério de parada físico, com convergência por
construção (dissipação ⇒ ponto fixo), que ACT/PonderNet (parada probabilística),
DEQ (solver implícito caro) e Geiping (r fixo/aleatório, sem parada adaptativa)
não têm.

### Mecânica

```
e        = embedding do prompt (re-injetado a cada passo — essencial, cf. Geiping)
z0       = e (+ ruído τ)        p0 = 0

força:   F(z) = f_θ(z, e) − z            # o bloco weight-tied "puxa" o latente
momento: p ← (1−γ)·p + η·F(z)            # γ = fricção (dissipação)
estado:  z ← z + η·p
energia: E = ½|p|² + ½|F(z)|²            # cinética + residual de ponto-fixo
parada:  ΔE < ε  ou  r = r_max
```

`f_θ` = um grupo de blocos Jamba (Mamba2 + atenção + FFN/KAN ternário).

### Botões cognitivos (já existem no H-TTT)

| Botão | Papel físico | Papel cognitivo |
|---|---|---|
| γ (fricção) | taxa de dissipação | orçamento de compute: alto = resposta rápida; baixo = explora mais |
| τ (temperatura) | ruído no fluxo | **self-consistency latente**: k trajetórias estocásticas do MESMO prefixo (só o core re-roda), voto por maioria/energia |
| r_max | teto de passos | profundidade máxima de pensamento |
| ε | estacionariedade | sensibilidade da parada |

## 2. Por que deve melhorar GENERALIZAÇÃO (o argumento científico)

Recorrência weight-tied aprende **algoritmos**, não memorização: Schwarzschild
et al. ("Can You Learn an Algorithm?") mostram extrapolação para instâncias
maiores/mais difíceis que o treino apenas iterando mais no teste; Geiping
confirma o princípio em escala de LLM. **Parâmetro compra memória; iteração
compra algoritmo.** Para um modelo edge de ~40M que não pode escalar parâmetros,
profundidade recorrente é a única alavanca de generalização que escala com
compute de inferência — e o corpo ternário torna cada iteração ~ADD/SUB
(quase grátis em energia no edge).

## 3. Honestidade sobre novidade (trabalho relacionado)

- **Energy Transformer** (Hoover et al.): forward como descida de energia — mas
  sem momento/fricção/temperatura, sem parada por estacionariedade, sem ternário.
- **DEQ**: ponto fixo implícito com solver — caro/instável; LHR é integração
  explícita estável com BPTT truncado.
- **Recurrent depth** (Geiping): unroll aleatório, sem parada adaptativa.
- **ACT/PonderNet**: parada probabilística aprendida, sem garantia.

**A novidade é a síntese**: integrador simplético dissipativo COMO mecanismo de
raciocínio latente, com parada por energia, sobre corpo ternário. As três
diferenciações fortes: convergência por construção (γ>0), self-consistency
latente via τ, e custo de iteração ternário.

## 4. Receita de treino (Geiping-style, adaptada)

- r ~ LogNormal(log r̄, σ) por batch (r̄≈4–8 no MVP)
- **BPTT truncado nos últimos k passos** (k≈4) — memória O(k), não O(r)
- supervisão profunda opcional (CE em saídas intermediárias, peso pequeno)
- z0 = e + ruído; e re-injetado a cada passo
- QAT/ternário inalterados (o core é BitLinear normal)

## 5. Sinergia com a infra existente

1. **CUDA Graphs**: core weight-tied + shapes estáticos ⇒ capturar 1 iteração e
   replay r vezes (launch overhead ~0). O **memory pool** (commit a0618ed, 3.9×)
   fornece exatamente os endereços estáveis exigidos pela captura.
2. **Ternário**: dentro do loop, BitLinear empacotado = ADD/SUB.
3. **KAN como potencial**: caminho KAN medido mais rápido/leve que FFN no GPU
   path; KANs representam bem funções suaves — o potencial U(z) é uma.
4. **Plumbing existente**: `JambaModel::run_reasoning_loop` / `forward_thought`
   já estão expostos nos bindings; o H-TTT já implementa momento/fricção/
   temperatura (apontado hoje para adaptação de pesos — repontar para o latente).

## 6. MVP (bounded, default OFF)

- Flag `use_latent_reasoner` em ModelConfig + knobs: `lhr_r_mean`, `lhr_r_max`,
  `lhr_friction`, `lhr_temperature`, `lhr_halt_eps`, `lhr_bptt_k`.
- Core = 1 grupo de blocos (ex.: os 2 últimos blocos Jamba antes do head).
- Treino: amostrar r, truncar BPTT (k passos) — usar o backward_external
  repetido sobre o grupo.
- Inferência: loop com parada por ΔE<ε; τ>0 habilita k trajetórias + voto.

### Gates (padrão ouro)
1. Paridade OFF: com flag OFF, byte-idêntico ao caminho atual (default preserva).
2. Sanidade ON: loss converge; sem NaN; `verify_determinism` com τ=0.
3. **Curva de pensamento**: acurácia/loss held-out monotônica em r (r_test >
   r_train ⇒ melhora em tarefas algorítmicas: adição multi-dígito, paridade,
   profundidade de parênteses) — ESTE é o gate científico do LHR.
4. Throughput: tok/s com r=1 ≈ baseline (overhead do loop ~0 com graphs).

## 7. Ideias-irmãs (registradas, prioridade menor)

- **HCB — Holographic Context Bundle**: SSA mantém sink+local exatos; o passado
  distante vira UMA superposição holográfica (HAM bind/bundle com permutação
  posicional; recuperação = unbind). Contexto arbitrário com memória O(1).
  Peças já existem (sparse_attention.cpp + holographic.cpp).
- **TTT² — Test-Time Topology**: adjacência do CHRASS atualizada Hebbianamente
  por co-ativação na sessão ⇒ o modelo reescreve a própria topologia conceitual
  durante a conversa (TTT para estrutura, não pesos).

## 8. Performance extrema — pendências priorizadas (com dado)

O diagnóstico (`NSOS_TRAIN_TIMING`, commit 348e512) mostrou fwd+bwd dominando o
step (loss-path ≈ 13ms). Alavancas restantes, em ordem:
1. **Fused Linear-CE (estilo Liger)**: nunca materializar logits
   `[B·T, vocab]` (~100MB/step no v11) — fundir lm_head+softmax+CE em chunks de
   vocabulário. Maior ganho único de memória+tempo do treino.
2. **CUDA Graphs no step** (pool já dá endereços estáveis).
3. CE batelado (1 kernel p/ o batch) + dispatch MoE device-side (block-sparse
   GEMM, já documentado em jamba.cpp como PR futuro).
