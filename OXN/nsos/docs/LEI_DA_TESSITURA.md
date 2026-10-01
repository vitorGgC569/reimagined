# A Lei da Tessitura — o Tear de Três Fios
### Reconstrução da matemática do aprendizado a partir do que só este projeto mediu.
**Status:** teoria original pré-registrada. Sem citações por design (§8 declara a
fronteira com honestidade). Toda validação é falsificável no nosso rig.

---

## 0. O ponto de partida: parar de tratar os três eixos como três assuntos

A TEAR já dizia: tempo, profundidade e plasticidade são a MESMA recorrência
afim `z' = a⊙z + u` em eixos diferentes. A noite inteira de reflexão se reduziu
a uma pergunta que a TEAR fazia sem perceber:

> Se os três eixos são a mesma equação, **o que acontece com um bit de
> informação que precisa atravessar OS TRÊS EM SÉRIE para virar capacidade?**

Porque é isso que aprender É. Um padrão dos dados, com escala natural τ
(distância entre pista e uso), só vira capacidade se sobreviver à travessia:

```
dados ──[fio 1: TEMPO]──> representação no estado h
      ──[fio 2: PROFUNDIDADE]──> crédito chegando à camada certa (backward)
      ──[fio 3: PLASTICIDADE]──> retenção no peso até a próxima ocorrência
```

Cada fio é um produto de ganhos aleatórios (Kesten/Oseledets). Composição EM
SÉRIE de canais multiplicativos ⇒ **os espectros se multiplicam**:

```
Aprendibilidade(τ) ∝ μ_T(1/τ) · μ_D(ℓ(τ)) · μ_P(1/R(τ))
```

onde μ_T, μ_D, μ_P são as densidades de expoentes (taxas de e-folding) de cada
eixo, e R(τ) é o intervalo de reocorrência do padrão no treino. Um zero em
QUALQUER fio zera o produto. Esta é a Lei da Tessitura: **capacidade só se tece
onde os três fios se cruzam na mesma escala.**

## 1. O que nós já provamos sem saber que era isso

| Evidência nossa (medida) | Leitura pela Lei |
|---|---|
| A=ones → recall 12.5% (chance); A log-espaçado → 100% | μ_T era um **delta** (um único polo). O fio 1 não cobria τ da tarefa. Um espectro log-uniforme cobriu. |
| QAT descriticaliza (frac_g 0.976→0.214); controlador P4 segura a banda | μ_D deriva sob quantização; sem g≈1 o fio 2 rompe em profundidade. |
| Backward ×3.2/camada vs forward ×1.1; 32%/step de ruído de hardware no gradiente | O fio 2 tem espectro ASSIMÉTRICO: o crédito é caótico. Existe um **horizonte de crédito** d* = ln(1/ε)/ln(χ_b) ≈ 14 camadas em FP32 — abaixo disso, crédito limpo por step é ~zero. |
| O modelo aprendeu primeiro o próprio código-fonte | Captura por menor entropia = o fio 3 com UM único timescale consolida primeiro o que reocorre mais rápido e mais regular. |

Cada truque consagrado do campo é, sem que se diga, "tornar UM fio
log-uniforme": init espectral de SSMs (fio 1), residuais/normalizações e
criticalidade (fio 2), e... **o fio 3 não tem truque. Ninguém o construiu.**

## 2. A peça que falta (a que "nem nós sabíamos")

> **O otimizador é uma camada da rede.** Não um meta-processo fora dela: é o
> terceiro segmento do caminho do sinal, com sua própria recorrência
> `m' = β·m + (1−β)·g`, `w' = w − η·(...)` — e está construído, hoje, em TODO
> o campo, exatamente com o bug que nós consertamos no Mamba: **um único polo**
> (um β1, um β2, um η para todos os pesos). O Adam escalar é o `A=ones` do
> fio da plasticidade.

A consequência estrutural: a largura de banda de capacidade do sistema é a do
**fio gargalo**. Nosso v11 tem 64 canais de A por camada cobrindo τ∈[1,100]
(spread medido = 2.0 décadas) no fio 1 — e UM timescale efetivo no fio 3.
Estamos pagando parâmetros por uma tessitura de tempo que a plasticidade não
consegue tecer.

## 3. O teorema pequeno (e a prescrição que cai dele)

**Cobertura minimax de escalas.** Para cobrir [τ₁, τ₂] com N canais de 1ª
ordem (cada canal cobre [τᵢ/c, c·τᵢ]), a alocação que maximiza a pior
cobertura é única: **espaçamento geométrico** τᵢ = τ₁·(τ₂/τ₁)^((i−1)/(N−1)).
(Pigeonhole em escala log: qualquer outra alocação abre buraco.) Log-uniforme
não é "bom" — é ótimo. Aplicado por fio:

- Fio 1: re-deriva o nosso fix do A (e o porquê de ele ser tão decisivo).
- Fio 3 (**novo**): as constantes do otimizador devem formar uma escada
  geométrica — por banco de pesos ou por camada: β2ᵢ (e/ou ηᵢ) log-espaçados,
  cobrindo desde R_min (reocorrência mais curta no batch) até R_max (mais
  longa no curriculum). Implementação trivial no nosso otimizador fundido
  (tabela por-tensor: +1 coluna).
- Acoplamento (**novo**): os TRÊS intervalos de cobertura devem ser o MESMO
  intervalo — casados à distribuição de escalas da TAREFA. Desenho de
  arquitetura vira um problema de afinação: igualar tessituras.

## 4. Predições pré-registradas (cada uma mata a lei se falhar)

**P1 — O recall do fio 3 (o experimento-assinatura).** Mesma tarefa E3 que
validou o fio 1, mas agora com fio 1 PERFEITO (A log-espaçado) e variando só a
reocorrência R (nº de payloads / frequência de revisita) contra o horizonte do
otimizador H=1/(1−β2). Predição: acc colapsa quando R ≫ H mesmo com fio 1
ideal, e FLIPA reintroduzindo cobertura (β2 mais lento ou 2 bancos), a budget
idêntico. Uma variável, mesmo desenho do E3. *Mata a lei se* acc for
insensível a R/H.

**P2 — A superfície-produto.** Grade (banda de A) × (banda de β2) × τ da
tarefa: a acurácia deve ser ≈ produto/min de duas rampas — colapsa quando
QUALQUER banda exclui τ, com fronteiras em 1/τ (fator 2-3). Teorias de
capacidade pura não predizem essa fatoração. *Mata se* a superfície não
fatorar.

**P3 — BF16 come ~5 camadas de crédito.** Do horizonte d* = ln(1/ε)/ln(χ_b):
trocar FP32→BF16 (ε: 2⁻²⁴→2⁻⁸ efetivo) encurta o horizonte em
ln(2¹⁶)/ln(3.2) ≈ 9.5… recalculando com χ_b=3.2: Δd* = 16·ln2/1.163 ≈ 9.5 ⇒
com nosso χ_b medido, a previsão conservadora: **a coerência de réplica r_ℓ
em BF16 cruza 0.5 entre 4 e 6 camadas ANTES do que em FP32** (sonda de
réplica já existente, braços bf16 vs fp32). Ninguém modela mixed-precision
como PERDA DE PROFUNDIDADE DE CRÉDITO. *Mata se* r_ℓ(bf16) ≈ r_ℓ(fp32).

**P4 — O gargalo é o fio 3, não o 1.** No v11 atual, alargar μ_P (2 bancos de
β2: rápido/lento, metade dos tensores cada) ganha mais loss de validação do
que alargar μ_T (A de 2→3 décadas), a budget igual. *Mata se* o fio 1 ganhar.

**P5 — Avalanches de consolidação.** Flips ternários (eventos discretos do
fio 3 sob QAT) devem exibir estatística crítica (cauda pesada) exatamente
quando o controlador P4 segura g na banda — ligando os fios 2 e 3 numa única
observável que já sabemos contar (diff dos pesos empacotados).

## 5. Programa mínimo (tudo nos harnesses existentes)

1. P1/P2: `crit_experiments.py recall` + colunas por-tensor no otimizador
   fundido (1 sessão de T4).
2. P3: sonda de réplica (paridade v5) em dois braços de precisão (~10 min).
3. P4: A/B de 300 steps, 3 seeds.
4. P5: contador de flips por checkpoint (host, barato).
Ordem: P3 (mais barata e mais estranha) → P1 → P4 → P2 → P5.

## 6. O que a lei diz sobre AGI edge (MARCO UM)

Edge não compete em largura de fio por força bruta — compete em **afinação**:
medir a distribuição de escalas do ambiente do usuário (reocorrências reais)
e afinar as três tessituras NELA, em malha fechada (o padrão do controlador
P4, agora nos três eixos). Um modelo pequeno perfeitamente tecido na tessitura
do SEU mundo bate um grande desafinado — essa é a aposta matemática do projeto,
agora com lei, teorema e cinco experimentos.

## 7. Relação com o Espectro de Crédito

O Espectro de Crédito (doc anterior) é a METROLOGIA do fio 2. A Tessitura é a
lei dos três fios que o contém: SNR por camada é como se mede onde o fio 2
cobre; a lei diz o que fazer com os outros dois.

## 8. Fronteira honesta (sem citações, por instrução)

Existem isoladamente: espectros log na INIT de SSMs; criticalidade/borda do
caos em PROFUNDIDADE; lr por camada via norma; EMAs múltiplas em otimizadores
isolados como truque empírico. O que não conheço em lugar nenhum: (a) a
afirmação de que os três eixos compõem EM SÉRIE e portanto seus espectros se
MULTIPLICAM (lei do produto, P2); (b) o otimizador tratado como camada da
rede com a prescrição minimax de escada geométrica derivada (não tunada);
(c) precisão numérica como PROFUNDIDADE DE CRÉDITO (P3, com fórmula); (d) o
casamento das três tessituras à distribuição de escalas da tarefa como
princípio de design. Falsificação bem-vinda — o pré-registro acima fica de pé
como replicação independente se algo disso existir.

---
*Derivado de: E1-E5, o flip 0.125→1.0 do A-spectrum, frac_g 0.976→0.214 + P4,
χ_fwd≈1.1 / χ_bwd≈3.2 medidos, 32%/step de ruído de réplica, o modelo
recitando tensor.cpp, e a recorrência TEAR. Nenhuma execução nova foi
necessária para formular; cinco execuções baratas decidem.*
