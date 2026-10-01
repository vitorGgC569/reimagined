# TEAR — A Recorrência Triaxial
### "Uma lei, três relógios. O zoológico vira um organismo."

**Status:** proposta de arquitetura fundamental (pesquisa/incubação).
**Relação com docs anteriores:** OXTA-CRIT é a LEI (onde sentar no diagrama de
fases); Pêndulo e ENGRAMA eram sínteses de módulos — o TEAR os DISSOLVE: viram
casos especiais de um único primitivo. Este doc responde "qual é o problema da
estrutura inteira" em vez de remendar partes.

---

## 1. O diagnóstico do todo (o problema real)

O NSOS é uma FEDERAÇÃO de primitivos: atenção (recuperação por conteúdo),
Mamba (integração temporal), MoE (roteamento), KAN (ajuste suave), TTT
(plasticidade), CHRASS (estrutura relacional), HAM (binding), MCTS
(deliberação) — colados pelo fluxo residual e por uma única loss. Cada um tem
sua matemática, seus bugs, seus kernels, seus hiperparâmetros. O problema da
estrutura inteira não é nenhum componente: **é que não existe substrato comum**.
Retalhos, como o usuário nomeou.

O mainstream tem o problema oposto: UM primitivo (atenção+MLP) repetido —
uniforme porém rígido. A pergunta certa: existe um primitivo único do qual os
comportamentos do zoológico EMERGEM como regimes, em vez de serem módulos?

## 2. A entrelinha do código inteiro

Olhe cada componente do NSOS sem o nome dele. TODOS são a mesma equação — a
recorrência afim — aplicada a um eixo diferente:

```
                 z' = a ⊙ z + u(entrada)
```

| Componente | A mesma equação, em qual eixo |
|---|---|
| Mamba2 scan | h_t = a_t h_{t−1} + b_t x_t — eixo TEMPO (tokens) |
| Fluxo residual (12 camadas) | z_{l+1} = 1·z_l + F(z_l) — eixo PROFUNDIDADE |
| Pêndulo/assentamento | z ← z + η·p — eixo PROFUNDIDADE iterada |
| TTT Hamiltoniano | W' = (1−γ)W + ΔW — eixo PLASTICIDADE (pesos) |
| AdamW (m, v) | m' = β₁m + (1−β₁)g — eixo TREINO |
| HAM binding | a ⊙ z É o produto Hadamard — o GATE da própria equação |
| MoE/seletividade | escolher quais canais integram vs esquecem = modular a — ROTEAMENTO é o gate |
| Atenção linear | ≡ fast weights (Schlag et al. 2021) — eixo PLASTICIDADE |
| CHRASS | acoplamento lateral entre canais — a ESTRUTURA de vizinhança do gate |
| EMA/normas/healer | mesma recorrência, relógios lentos |

A análise de Kesten (OXTA-CRIT Lei 2) já provou nesse sistema que **a posição
de (λ, σ) dessa recorrência determina capacidade** (E3: chance→100%). Ou seja:
já temos a lei de controle do primitivo. Falta assumir o primitivo como ÚNICO.

## 3. A invenção: TEAR

**Um único operador G (a "célula"), com pesos compartilhados, propagado em TRÊS
eixos ortogonais — cada eixo posicionado na sua linha crítica:**

1. **Relógio do TEMPO (sequência):** G varre os tokens como scan seletivo, com
   espectro de timescales log-espaçado/aprendido (λ, σ, κ por canal — Lei 2).
   *Era: Mamba.*
2. **Relógio do PENSAMENTO (profundidade):** o MESMO G é iterado r vezes
   (weight-tied) com parada por assentamento — profundidade vira computação
   adaptativa, não pilha de módulos distintos. *Era: as 12 camadas + Pêndulo.*
3. **Relógio da PLASTICIDADE (pesos rápidos):** o resíduo lento de G acumula em
   fast-weights ΔW pela MESMA recorrência (decaimento + outer-product),
   lidos no passo seguinte. Pela equivalência fast-weights ≡ atenção linear
   (Schlag/Schmidhuber), **este eixo SUBSUME a atenção** — atenção deixa de ser
   módulo e vira o relógio lento do mesmo tecido. *Era: atenção + TTT + ENGRAMA.*

A nonlinearidade interna da célula pode ser KAN (medida mais leve/rápida no GPU
path); o acoplamento lateral dos canais é o lugar natural do CHRASS; o gate
Hadamard é o lugar natural do HAM. Nada é jogado fora — tudo é **reabsorvido
como aspecto do primitivo**, não como módulo.

**A lei de colocação:** cada eixo tem seu (λ, σ) — tempo na linha marginal com
espectro κ; pensamento perto do slowing-down crítico; plasticidade com
decaimento lento e escrita gated por surpresa. O probe e o controlador já
construídos (criticality_probe, NSOS_CRIT_REG) governam OS TRÊS com a mesma
matemática. OXTA-CRIT deixa de ser diagnóstico e vira o **sistema operacional
de fases** do organismo.

## 4. Por que isto resolve o problema da estrutura INTEIRA

- **Colapso de parâmetros:** pesos amarrados nos 3 eixos ⇒ os 40M de params
  passam a codificar HABILIDADE (o operador); capacidade vem de ITERAÇÃO
  (pensamento) e ACUMULAÇÃO (plasticidade). Já provamos no E3 e na literatura
  (Schwarzschild; Geiping) que recorrência compra generalização que parâmetro
  não compra nesta escala. O déficit Chinchilla muda de natureza.
- **Um conjunto de kernels, um diagrama de fases, um audit:** em vez de 8
  matemáticas, uma. O custo de manutenção/kernels/auditoria cai uma ordem.
- **Edge nativo:** célula pequena ternária iterada = arquivo pequeno, iterações
  ADD/SUB baratas, "pensar mais" e "lembrar" sem crescer o binário.
- **Identidade:** nenhum lab está propondo "um operador, três relógios
  criticamente posicionados". É uma TESE de arquitetura, não um tweak.

## 5. Honestidade sobre novidade e riscos

**Peças conhecidas separadamente:** weight-tying em profundidade (Universal
Transformer; Geiping), fast weights ≡ atenção linear (Schmidhuber 1992; Schlag
2021), TTT como aprendizado em teste (Sun 2024), scan seletivo (Mamba). 
**A claim nova:** a UNIFICAÇÃO triaxial — um único operador compartilhado nos
três eixos, com colocação crítica por eixo como lei de controle — e a
reabsorção dos módulos do NSOS como aspectos. Não conhecemos arquitetura com
essa forma.

**Riscos reais:** (i) perda de expressividade pelo tying (mitigação: célula com
modulação por eixo — FiLM/condicionamento barato por relógio); (ii)
estabilidade de treino com 3 recorrências aninhadas (mitigação: BPTT truncado
por eixo + controlador de criticalidade — já validado no E5); (iii) capacidade
dos fast-weights (limite conhecido de memórias associativas lineares;
mitigação: gate de surpresa + decaimento Kesten com cauda pesada).

## 6. MVP falsificável (1050 Ti, pequeno)

**TEAR-célula v0:** UMA célula (gate seletivo + mistura de canais + KAN/FFN
pequena), 3 eixos ligados: scan no tempo; r=4–8 iterações weight-tied na
profundidade; fast-weights com decaimento por sequência. Comparar com baseline
mamba_small **com mesmos parâmetros TOTAIS** (a célula pode ser maior, já que
não há 12 pilhas):

Gates:
1. Suíte recall (E3) e tarefas algorítmicas: TEAR ≥ baseline param-matched.
2. **Extrapolação de iteração**: treinar com r=4, avaliar com r=8/16 — acurácia
   sobe (assinatura de algoritmo aprendido).
3. **ICL sem atenção**: tarefa de regressão/associação in-context resolvida só
   pelo eixo de plasticidade (predição da equivalência de Schlag).
4. Paridade de infra: determinismo; criticality_probe nos 3 eixos dentro das
   bandas.

Se (1)-(3) passarem em pequeno, o TEAR vira o candidato a v12 da arquitetura —
e o NSOS deixa de ser uma federação para ser um organismo.
