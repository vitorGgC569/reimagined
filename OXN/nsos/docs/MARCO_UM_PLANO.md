# MARCO UM — o primeiro passo a uma AGI de borda
### Programa de pesquisa do organismo persistente (sucessor do Marco Zero)

**Status:** plano de pesquisa pré-registrado. Nenhuma claim de AGI: a claim é
dar o primeiro passo MENSURÁVEL nas capacidades que escala não compra —
persistência, aprendizado contínuo, autodireção — no único regime onde elas são
nativas. Tudo falsificável; cada experimento cabe no hardware disponível
(GTX 1050 Ti local, T4 Colab, futura RTX 5060 Ti).

---

## 0. A tese (a inversão)

Datacenter = oráculo sem estado: nasce treinado, responde, esquece, morre a
cada request. As lacunas reais rumo a AGI — aprender continuamente sem
esquecer, manter identidade, intervir causalmente no ambiente, escolher o que
aprender, consolidar offline — exigem uma EXISTÊNCIA: tempo contínuo, um
ambiente consistente, escassez de energia, sono. Isso é o regime de borda.
**Edge não é a versão fraca do datacenter; é o habitat dos ingredientes que
faltam.** A escassez energética é a pressão seletiva que criou a inteligência
biológica — nós a tratamos como recurso de design, não como limitação.

## 1. O organismo (síntese do que já validamos + o que falta)

Fundações já estabelecidas NESTE repositório:
- **OXTA-CRIT** (lei de fases): espectro de timescales determina capacidade
  (E3: chance→100%); QAT descritifica e o controlador corrige (E4/E5, causal).
- **TEAR** (o primitivo): todo componente é a recorrência afim z'=a⊙z+u; um
  operador, três relógios (tempo/pensamento/plasticidade), criticamente
  posicionados.
- **ENGRAMA** (memória): ler=assentar; escrever=surpresa (energia residual).
- Infra: pool 3.9×, BF16, probe de fase, controlador, T4 pipeline.

O que o MARCO UM adiciona — os QUATRO LOOPS que transformam função em agente:

### Loop 1 — O quarto relógio: SONO ("Sono Crítico")
Biologia tem 4 escalas: atividade neural, assentamento, plasticidade rápida e
**consolidação offline** (replay hipocampo→córtex). Edge tem sono de graça: o
dispositivo carrega à noite. Proposta: durante idle, REPLAY priorizado por
surpresa transfere o que os fast-weights/slots aprenderam de dia para os pesos
ternários lentos — e a consolidação É annealing do retículo de spins
(Blume-Capel) com o controlador de criticalidade mantendo a borda. Ninguém fez
"consolidação por annealing crítico em LLM ternário". Ataca o problema central
do aprendizado contínuo (esquecimento catastrófico) com separação fast/slow:
de dia só os fast-weights arriscam; de noite os lentos mudam sob proteção de
replay.

### Loop 2 — A moeda única: SURPRESA
A energia residual do assentamento (que o TEAR computa de graça) vira a moeda
de QUATRO decisões: (i) quanto pensar (parada), (ii) o que escrever na memória,
(iii) o que repetir no sono (prioridade de replay), (iv) QUANDO aprender
(gating de update: não aprender o esperado — eficiência por joule). Free-energy
operacionalizado barato: um escalar, quatro decisões.

### Loop 3 — NEUROMODULAÇÃO DE FASE
O sistema regula a própria posição no diagrama de fases conforme o contexto,
como ACh/NE regulam ganho cortical: rotina → subcrítico (barato, confiável);
surpresa alta → crítico (máxima suscetibilidade/aprendizado); geração/exploração
→ levemente supercrítico (diversidade). Os atuadores já existem (γ, τ, alvos do
controlador); o sensor já existe (probe). Ninguém opera um LLM como um sistema
que dirige a própria fase.

### Loop 4 — CURIOSIDADE (seleção autônoma do que aprender)
Com déficit Chinchilla de 40×, a pergunta certa não é "mais tokens" e sim
"QUAIS tokens". Currículo fechado em loop: priorizar itens por progresso de
aprendizado estimado (Oudeyer), não novidade crua. No edge, cada correção do
usuário é uma INTERVENÇÃO causal rotulada — dado que datacenter não tem.

## 2. As perguntas de pesquisa (ninguém respondeu; todas falsificáveis)

**RQ1 — Um operador basta?** TEAR-célula triaxial param-matched vs pilha.
Gates: suíte recall/algorítmica ≥ baseline; EXTRAPOLAÇÃO de iteração (treina
r=4, avalia r=16: acurácia sobe = algoritmo aprendido); ICL SEM atenção (eixo
de plasticidade resolve associação in-context — predição da equivalência
fast-weights≡atenção linear). *Kill:* se perder nos 3 gates com modulação por
eixo já aplicada → tying morre, relógios viram módulos (e registramos).

**RQ2 — Sono crítico evita esquecimento?** Protocolo dia/noite em toy:
ensinar N fatos via fast-weights (dia) → consolidar com replay+annealing+
controlador (noite) → medir retenção dos fatos E perplexidade base.
*Predições:* retenção ≥80% com queda de base ≤2%; SEM o controlador de
criticalidade, esquecimento significativamente maior (liga OXTA-CRIT a
continual learning — ciência nova). *Kill:* retenção <50% após 3 iterações de
design → consolidação volta a fine-tune com replay clássico.

**RQ3 — Uma moeda governa quatro decisões?** (a) energia de assentamento
correlaciona com dificuldade/loss por item (P5 pré-registrada); (b) gating de
update por surpresa: mesma stream, ≥30% updates pulados com held-out igual ou
melhor (aprendizado por joule). *Kill:* correlação ~0 → surpresa não é moeda;
cada decisão precisa de sinal próprio.

**RQ4 — Curiosidade compra dados?** Mesmo budget de tokens, ordem escolhida
por progresso-de-aprendizado vs aleatória. *Predição:* held-out melhor com
mesmos tokens (eficiência amostral ataca Chinchilla pelo lado do dado).
*Kill:* sem ganho em 2 currículos distintos → seleção não paga nesta escala.

**RQ0 (em voo):** espectro κ transfere do toy para linguagem — é o run
t4_alog atual vs v11 (held-out de fases longas).

## 3. O benchmark que não existe: CEI (Continual Edge Intelligence)
Nenhum benchmark mede inteligência PERSISTENTE no dispositivo. Definimos:
UMA instância, avaliada ao longo de dias, em quatro eixos:
1. **Retenção**: fatos ensinados no dia D respondidos em D+k.
2. **Não-esquecimento**: perplexidade/skills base não degradam (±2%).
3. **Adaptação**: erro no idioma/correções do usuário cai com o tempo.
4. **Energia**: tudo por joule (a métrica de borda; ternário compete onde
   ninguém mede).
CEI-mini (toy, 1050 Ti) primeiro; CEI-7d (piloto de 7 dias) depois.
Flag-plant: publicar o harness é ocupar um território vazio.

## 4. Marcos e gates (dimensionados ao hardware)

- **M1 (sem. 1-2, 1050 Ti):** TEAR-célula v0 + 3 gates do RQ1.
- **M2 (sem. 2-4, 1050 Ti):** Vigília/Sono v0 + CEI-mini (RQ2, RQ3a).
- **M3 (sem. 3-5, T4):** RQ3b/RQ4 no pipeline de linguagem; neuromodulação de
  fase v0 (3 modos chaveados por percentil de surpresa).
- **M4 (sem. 5-8, T4):** candidato v12 = organismo TEAR pequeno treinado com o
  SO de fases; comparar v11 em held-out + longo alcance + CEI-mini.
- **M5 (com a 5060 Ti):** PILOTO DE 7 DIAS — a mesma instância, uma semana,
  aprendendo todo dia, dormindo toda noite, CEI diário. A demo que nenhum lab
  tem: "o mesmo modelo, sete dias, sem esquecer, no seu PC".

**Sucesso em 30 dias:** M1+M2 fechados (ou kills documentados), CEI-mini
existindo, retenção≥80%/esquecimento≤2% no toy.
**Sucesso em 90 dias:** v12 ≥ v11 em held-out E passando CEI; whitepaper
"Criticality-governed continual edge intelligence" com TODAS as predições
pré-registradas e seus resultados (inclusive os negativos).

## 5. Disciplina do programa (o que nos protege de nós mesmos)
- Toda predição é pré-registrada ANTES do experimento (como P1-P5/E1-E5).
- Todo experimento tem kill-criterion explícito; negativo também é resultado
  e é commitado.
- Um run de cada vez muda UMA variável (lição do t4_alog).
- Honestidade de novidade: peças conhecidas citadas (Universal Transformer,
  fast-weights/Schlag, TTT/Sun, replay/CLS theory de McClelland, free energy/
  Friston, curiosidade/Oudeyer). O inédito é a síntese operada por UMA lei de
  fases num corpo ternário persistente — e o CEI como régua.
- O que deliberadamente NÃO fazemos: perseguir escala de datacenter; claims de
  AGI; benchmark-chasing sem joule no denominador.

## 6. Em uma frase
> Marco Zero construiu o corpo. **Marco Um dá a ele uma vida**: dias para agir
> e se surpreender, noites para consolidar, uma fase para regular e uma régua
> (CEI) para provar — o primeiro organismo de inteligência persistente de
> borda, medido em capacidade por joule por dia.
