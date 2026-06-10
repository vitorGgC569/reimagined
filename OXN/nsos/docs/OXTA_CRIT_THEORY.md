# OXTA-CRIT — O Princípio da Criticalidade
### "A inteligência não está nos parâmetros nem nos dados. Está na FASE."

**Status:** teoria / pesquisa (incubação). Mensurável com a infra existente
(layer_audit, parameters(), Tensor.numpy()). Não é claim de produto.

---

## 0. A entrelinha que ninguém escreveu

Dois objetos do NSOS são tratados pelo mundo como aproximações de engenharia.
Lidos com olhos de física estatística, eles são outra coisa:

1. **Um peso ternário {−1, 0, +1} não é um "número comprimido" — é um SPIN-1
   com vacância.** Uma matriz ternária é um grafo assinado: excitação (+1),
   inibição (−1), ausência (0). O sistema de spins com três estados e controle
   de densidade de vacâncias é o **modelo de Blume-Capel**:
   `H = −J Σ sᵢsⱼ + D Σ sᵢ²`, onde D (custo de ser ±1 vs 0) controla a
   esparsidade p₀ — e cujo diagrama de fases contém um **ponto tricrítico**.
   ⇒ A esparsidade do BitNet não é um detalhe de compressão: é a TEMPERATURA
   EFETIVA do sistema de pesos. QAT é annealing num diagrama de fases.

2. **O scan seletivo `h_t = a_t·h_{t−1} + b_t·x_t` não é decaimento exponencial
   — é um PROCESSO DE KESTEN** (recorrência multiplicativa-aditiva com a_t
   flutuante, pois a seletividade torna a_t dependente do dado). Teorema de
   Kesten: com λ = E[log a] < 0 e flutuação σ² = Var[log a] > 0, a distribuição
   estacionária do estado tem **cauda de lei de potência** P(h>X) ~ X^(−κ),
   com κ resolvendo E[a^κ] = 1 (κ ≈ 2|λ|/σ² no regime gaussiano).
   ⇒ Memória do Mamba seletivo NÃO precisa ser exponencial: perto da linha
   marginal (λ→0⁻) com seletividade flutuante, a retenção é **livre de escala**
   — esquecimento rápido do comum, retenção longa do raro — sem nenhum
   mecanismo extra de longo prazo.

## 1. O Princípio (a lei única)

> **Computação máxima vive na borda entre fases.** Treinar o modelo PARA a
> fase crítica — e mantê-lo lá — em três setores ao mesmo tempo:

### Lei 1 — Espaço (Body): manifold crítico ternário
Campo médio da propagação de sinal para pesos ternários com fração de zeros p₀,
escala α (weight_scale/absmean), fan-in N:

```
ganho de ramo:  g = α² · N · (1−p₀) · E[φ'(h)²]
manifold crítico (rede plana):  g = 1  ⇒  α_c ∝ 1/√(N(1−p₀))
```

**Consequência nova:** esparsidade e escala devem COVARIAR — quando o QAT
esparsifica (p₀↑), α deve subir ∝ 1/√(1−p₀) para a camada não "congelar"
(sinal morre) nem "ferver" (explode).
**Poder explicativo:** instabilidade de QAT = *descritificação* — o ruído de
quantização chuta camadas para fora do manifold crítico; os spikes de loss ao
ligar QAT são a assinatura. O conserto não é "lr menor": é regularizar g→g*.
**Caveat honesto (residual+norm):** em blocos residuais o critério plano g=1
muda; o alvo correto é ganho de ramo uniforme e O(1/L)-controlado em
profundidade (Σ log(1+g_l) limitado). O probe mede g por camada; o alvo fino é
calibrado empiricamente.

### Lei 2 — Tempo (Mind): espectro de memória de Kesten
Por canal/cabeça do scan: a_t = exp(−softplus(dt_t)·A). Defina
λ = E[log a_t], σ² = Var[log a_t], **κ = 2|λ|/σ²**.

- κ grande ⇒ memória curta, cauda leve (esquece tudo rápido).
- κ→0⁺ (λ→0⁻ com σ²>0) ⇒ **memória livre de escala** (cauda pesada — retenções
  raras e muito longas emergem do próprio processo).

**Dial novo:** treinar/regularizar um ESPECTRO de κ pelas cabeças — hierarquia
de escalas de tempo (como a hierarquia de timescales corticais), em vez do
decaimento uniforme de hoje. Long-context sem pagar atenção densa.

### Lei 3 — Dinâmica (Spirit): assentamento em slowing-down crítico
No core de assentamento (ENGRAMA/Pêndulo v2, energia Hopfield), operar o ganho
do loop perto do ponto de critical slowing down: tempo de relaxação cresce com
a "dificuldade" do input (energia residual inicial). **Compute adaptativo
emerge da física**: inputs fáceis assentam em 1-2 passos, difíceis demoram —
sem controlador aprendido. Parada por ΔE<ε (já no design ENGRAMA).

### Unificação com a marca
Body = criticalidade espacial (spins ternários) · Mind = criticalidade temporal
(Kesten) · Spirit = criticalidade dinâmica (assentamento). Uma lei, três
setores. Suporte biológico: hipótese do cérebro crítico (avalanches neuronais,
Beggs & Plenz 2003; maximização de faixa dinâmica em σ_branch=1, Kinouchi &
Copelli 2006). O NSOS seria o primeiro stack de LLM **projetado e AUDITADO para
operar em criticalidade** — a camada industrial (layer_audit) ganha um gate
físico: expoente de avalanche / ganhos de ramo / espectro κ.

## 2. Falsificabilidade (o que torna isto ciência, não estética)

Previsões testáveis com a infra atual:
P1. Camadas com g longe do alvo correlacionam com as patologias que o
    layer_audit já regista (saturação/silêncio).
P2. Os spikes de loss ao ativar QAT progressivo coincidem com saltos de p₀ SEM
    a covariação compensatória de α (descritificação mensurável).
P3. Cabeças com κ alto falham probes de retenção longa; um espectro de κ
    melhora held-out de long-context sem custo de atenção.
P4. Com o regularizador de criticalidade (g→g*), a mesma config treina com lr
    maior sem divergir (susceptibilidade controlada).
P5. No assentamento: tempo-até-ΔE<ε correlaciona com dificuldade do item
    (proxy: loss por amostra) — compute adaptativo emergente.

## 3. Honestidade sobre o que já existe

- Edge of chaos / propagação de sinal em campo médio: Poole et al. 2016,
  Schoenholz et al. 2017 (para pesos GAUSSIANOS; dropout análogo).
- Caudas pesadas via ruído multiplicativo/Kesten em SGD: Hodgkinson & Mahoney,
  Gürbüzbalaban et al. (sobre DINÂMICA de treino, não sobre estado de SSM).
- Criticalidade no cérebro: Beggs-Plenz; Kinouchi-Copelli.
- Hopfield/energia: Ramsauer et al.; Energy Transformer.

**O que é nosso:** (i) o mapa ternário↔Blume-Capel com esparsidade-como-
temperatura e a LEI DE COVARIAÇÃO α²(1−p₀)≈const como regra de controle do QAT;
(ii) o scan seletivo lido como Kesten com κ por cabeça como DIAL DE ESPECTRO DE
MEMÓRIA; (iii) criticalidade como GATE INDUSTRIAL auditado; (iv) a unificação
dos três setores num princípio só, num corpo ternário edge. Não conhecemos
trabalho que faça qualquer um dos quatro.

## 4. Programa mínimo (mensurável já)

- **Probe v1 (`scripts/criticality_probe.py`):** por camada BitLinear — p₀
  (regra absmean do BitNet), γ, N, ganho de ramo ternário g_tern e contínuo
  g_cont, amplificação acumulada em profundidade; por camada Mamba — espectro
  de timescales 1/A (estático). Roda em CPU/GPU sobre modelo inicializado ou
  checkpoint. (v2: tap de dt para λ, σ, κ reais por cabeça.)
- **Regularizador (Lei 1):** termo tipo qat_regularization puxando g_l → g*
  via ajuste de α (mesma plumbing do trainer; sem kernel novo).
- **Gate de avalanche (Lei 1/3):** layer_audit ganha fit de expoente de
  avalanche em ativações limiarizadas.
- **Espectro κ (Lei 2):** inicialização/regularização de A e dt-proj para
  distribuir 1/A em escala log (hierarquia); depois κ real com o tap v2.

## 5. O diferencial Oxta, em uma frase

> Não vendemos um modelo maior. **Vendemos um modelo na fase certa** — pesos na
> borda da ordem, memória na borda do esquecimento, pensamento na borda do
> assentamento — auditado fisicamente, num corpo ternário que cabe no edge.
