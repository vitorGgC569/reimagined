# Tessitura, pós-fogo — o que sobreviveu a dois rounds adversariais
**Método:** a "Lei da Tessitura" (v1) foi submetida a dois rounds de revisão
adversarial máxima (árbitro independente, regras rígidas: nota por claim,
contraexemplo obrigatório, colisões nomeadas). Este doc registra o que MORREU,
o que foi REBAIXADO e o que ficou de pé como **genuinamente novo e testável**.
A v1 (`LEI_DA_TESSITURA.md`) permanece como registro histórico, agora
FALSIFICADA-COMO-LEI pelo processo que nós mesmos pedimos.

## 1. O que morreu (e por quê — colisões nomeadas pelo árbitro)

- **A lei do produto μ_T·μ_D·μ_P**: erro de tipo; a rede é SOMA de caminhos
  paralelos (a forma correta é soma-de-produtos, que não fatoriza). Conjunção
  fwd/bwd já é o programa edge-of-chaos (Poole/Schoenholz/Pennington).
- **"Otimizador = camada; escada geométrica de β's"**: já existe (AggMo,
  AdEMAMix, TD(λ), RLS, Benna-Fusi). Resta no máximo aplicação comparativa.
- **"Precisão = profundidade de crédito" com parede d\***: confundia variância
  (média zero, se suprime com steps — McCandlish) com viés (a razão real de
  loss scaling/stochastic rounding existirem); nossas duas medições de χ_b
  eram estimadores DIFERENTES (Lyapunov do ruído vs atenuação do sinal) e
  inconsistentes entre si. Metz 2021 tem a fórmula do horizonte no eixo tempo.
- **"Teorema da Assimetria" (T_P fatora por ser global)**: Adam é
  por-coordenada (T_P é matriz estado-dependente, não prefator); decay
  por-grupo/µP já localizam T_P; e Continual Backprop (Dohare 2024) cura
  plasticidade com resets LOCAIS ⇒ plasticidade É roteável. Ou falso, ou
  tautologia.
- **"Consolidação ternária GRATUITA"** como mecanismo geral: é a tese do Bop
  (Helwegen 2019: latente = inércia, flip por limiar de EMA), e Laborieux &
  Ernoult 2021 mostram que binarização sozinha NÃO protege — precisa
  metaplasticidade explícita. E o enigma se resolve contra nós: o limiar que
  absorve ruído absorve também sinal pequeno (por isso QAT degrada).

## 2. O que SOBREVIVEU (veredito do árbitro: "plausivelmente publicável-como-novo")

### S1 — P6b: a lei de escala do esquecimento ternário  ★ a jóia
**Claim:** em rede ternária sob QAT com tarefas intercaladas, a taxa de
esquecimento escala com a razão **excursão/limiar = η·σ_g/Δ** (passo do
random-walk de interferência sobre o limiar de quantização). Não reivindicada
por Helwegen nem Laborieux (que não medem lei de escala).
**Protocolo endurecido pelas exigências do árbitro:** (i) comparar FP32 vs
ternário **a plasticidade casada** (curva de aquisição da tarefa B igualada —
senão "menos esquecimento" se compra congelando a rede); (ii) variar η em ≥3
níveis e Δ em ≥2 para traçar a lei; (iii) sem função metaplástica explícita —
se precisar de uma, o claim "emergente" cai e dizemos isso.

### S2 — Metrologia do expoente de ruído-de-precisão por camada
**Claim:** réplicas pareadas re-sincronizadas (mesmo estado, 1 step, comparar)
medem λ_noise por camada — o expoente do canal de ruído de
hardware/precisão, separado de λ_sig (atenuação do gradiente médio, que é
Schoenholz e não é nosso). **Controle obrigatório (exigência F5):** medir a
variância de minibatch lado a lado e provar que λ_noise não é dominado por
ela — se for, o instrumento caracteriza canal subdominante e vira nota de
rodapé. Honestidade adicional: no regime d·gap grande o "multiplicador de
steps" e^{2·gap·d} é parede na prática (e^{16} ≈ 9e6×) — a distinção
parede/taxa só vale nas profundidades onde o expoente medido a mantém O(1-10).

### S3 — Controle de criticalidade em malha fechada DURANTE QAT progressivo
Round 1, §E2: sem colisão direta encontrada (GraSP/SynFlow/LSUV agem em INIT,
não em malha fechada durante quantização progressiva). Já validado por nós
(E4/E5). Pendência interna real (objeção A2): o controlador segura o g do
FORWARD; medir o que acontece com o expoente backward sob o controlador.

### S4 — Avalanches de flips ternários como observável de criticalidade
Round 1, §E1: sem colisão direta (vizinhos: Helwegen, SOC/Sethna). Com rigor
obrigatório: definição pré-registrada de avalanche, teste Clauset-Shalizi-
Newman com ≥3 décadas, nulo lognormal, e DOIS controles (g fora da banda ⇒
cauda pesada espúria por Simsekli; g na banda com LR→0 ⇒ congelamento).

### S5 — P4 comparativo (espectro do otimizador vs espectro de A, budget igual)
Qualitativamente antecipado por AdEMAMix; o comparativo direto com análise de
poder (ICs, >3 seeds) é nosso.

## 3. Programa experimental final (ordem, tudo no rig atual)

1. **S2-controle** (mais barato, decide o instrumento): réplicas pareadas vs
   variância de minibatch, por camada, FP32 e BF16 — 1 sessão T4.
2. **S1/P6b** (a jóia): grade η×Δ, plasticidade casada, FP32 vs ternário —
   usa o harness de tarefas intercaladas + contador de flips.
3. **S4/P5'**: estatística CSN dos flips com os dois controles — mesmo run de S1.
4. **S3**: telemetria do expoente backward sob o controlador — piggyback em
   qualquer treino.
5. **S5/P4**: por último (mais caro, precisa de poder estatístico).

## 4. Related work (a exigência F10 — o "sem citações por design" morreu aqui)
S4D/LRU/chrono-init (fio tempo); Poole 2016, Schoenholz 2017, Pennington 2017,
Yang & Schoenholz 2017 (edge of chaos/isometria); Balduzzi 2017 (shattered
gradients); Metz 2021 (horizonte de caos); Micikevicius 2017, Gupta 2015
(precisão: viés vs variância); McCandlish 2018 (noise scale); AggMo 2018,
AdEMAMix 2024, TD(λ), RLS, SAOL 2015, Benna-Fusi 2016 (multi-timescale);
Helwegen 2019 (Bop), Laborieux & Ernoult 2021 (metaplasticidade BNN),
Courbariaux 2015 (latente-acumulador); Dohare 2024, Lyle 2023, Sokar 2023,
Nikishin 2022 (plasticidade); µP (Yang & Hu); Zhang 2017 (memorização);
Simsekli 2019 (caudas pesadas); Carlini/Kandpal (memorização por duplicação);
Clauset-Shalizi-Newman (rigor de lei de potência).

## 5. A lição de método (vale mais que a v1 inteira)
Duas noites, duas teorias grandiosas, dois fuzilamentos — e o que restou são
**quatro alvos pequenos, afiados, NOSSOS e mensuráveis nesta semana** (S1-S4),
cada um com controles que o árbitro desenhou tentando matá-los. É exatamente
assim que o MARCO UM deve operar: imaginação no escuro, fogo adversarial,
e só o que sobrevive entra no rig.
