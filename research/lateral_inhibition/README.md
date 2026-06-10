# Lateral Inhibition — forward-only learning (módulo de validação)

> Módulo **à parte**, autocontido. **Não** está integrado ao NSOS (`OXN/nsos`).
> Depende apenas da biblioteca padrão de C++. Objetivo: validar uma ideia, não
> entregar produção.

## A ideia (do Oxta)

> *"O gato, ao caçar, inibe outros neurônios e foca somente naquilo."*

Isso tem nome em neurociência: **inibição lateral** / *winner-take-all*. O
neurônio mais ativado **suprime** os vizinhos. Combinada com uma regra de
aprendizado **local** (o vencedor se aproxima do estímulo), ela produz uma
camada que **aprende características sem backpropagation**:

- sem gradiente global, sem grafo, sem guardar ativações;
- **um único forward por amostra** (nativo de CPU / edge, contínuo);
- os pesos **se especializam apenas por causa da inibição** — sem ela, todos
  colapsam para o mesmo protótipo e nada é aprendido.

É exatamente a fuga da "dependência de datacenter": o que torna o treino caro
é o backprop (memória de ativações + passo global). Aprendizado local +
inibição treina na borda, em streaming.

## O que está aqui

| Arquivo | Papel |
|---|---|
| `lateral_inhibition.h` | A camada. `activate` (cosseno) → `inhibit` (top-k) → `learn` (regra competitiva local) → `step`. `k=1` = inibição total (WTA); `k=n_neurons` = inibição desligada. |
| `demo.cpp` | Gera 6 clusters na esfera unitária e **compara inibição ON vs OFF**, mede acurácia de um *readout* não supervisionado, e testa **adaptação online a drift**. |
| `build_and_run.bat` | Compila standalone com MSVC e roda. |

## Como rodar

```bat
cd research\lateral_inhibition
build_and_run.bat
```

(ou manualmente, com o ambiente do VS carregado: `cl /O2 /EHsc /std:c++17 demo.cpp /Fe:demo.exe && demo.exe`)

## Resultado medido (MSVC, /O2)

```
data: 1320 points, dim=16, 6 clusters | neurons=12  chance=0.167

[1] Lateral inhibition ON  (k=1, winner-take-all)
    => quant_err=0.3087  active=12  readout_acc=0.949  (~1.9M samples/s)

[2] Lateral inhibition OFF (k=12, no competition)
    => quant_err=0.7726  active=1   readout_acc=0.167   <- colapso total

[3] Drift / continual adaptation (no retrain, no backprop)
    quant_err on shifted data: 0.6146 -> 0.3247  (adapta online)

VERDICT:
  forward-only learning works ........ YES (readout 0.949 vs chance 0.167)
  lateral inhibition is THE cause .... YES (acc 0.949 vs 0.167; qerr 0.309 vs 0.773)
  continual on-device adaptation ..... YES
```

### Como ler isso

- **Aprende sem backprop.** Um *readout* (rotular cada neurônio pela classe
  majoritária que o ativa) atinge **94,9%** num problema de 6 classes onde o
  acaso é 16,7%. As features emergiram só com regra local.
- **A inibição é *a* causa.** Mesma camada, mesmos dados, mesma regra — só
  muda `k`. Com inibição (`k=1`): 12 neurônios ativos, erro 0,31, 95%. Sem
  inibição (`k=12`): **1** neurônio ativo (todos colapsaram no mesmo protótipo
  médio), erro 0,77, acurácia = acaso. A inibição não *ajuda* — ela é o que
  faz **existir** aprendizado.
- **Adapta na borda.** Deslocado o mundo (clusters novos), a mesma camada se
  reorganiza online (0,61 → 0,32) sem retreino e sem backprop. É o caso de uso
  de adaptação contínua em dispositivo.

## Limites honestos (o que isto NÃO prova)

- É *clustering* competitivo (k-means esférico online com WTA), validado como
  **mecanismo**. Não é, ainda, um substituto do backprop para uma rede
  profunda de propósito geral.
- Empilhar várias dessas camadas e aprender features **hierárquicas** úteis
  para uma tarefa de linguagem é a pergunta aberta — e o próximo passo se
  quisermos levar isto ao NSOS.
- A regra é puramente Hebbiana/competitiva; não há sinal de erro de tarefa.
  Para tarefas supervisionadas, o caminho é um *readout* treinado no topo das
  features (ou inibição + um sinal de modulação local).

## Próximos passos (se valer a pena evoluir)

1. **Profundidade:** empilhar 2–3 camadas, cada uma aprendendo sobre as
   ativações esparsas da anterior; medir se features compõem.
2. **Esparsidade temporal:** janela de inibição ao longo do tempo (o "foco"
   do gato persiste), aproximando de *predictive coding* local.
3. **Ponte com NSOS:** usar uma camada dessas como *front-end* de features
   que alimenta o BitLinear 1.58-bit — treino do front-end sem backprop, na
   CPU/edge, e só o readout final ajustado. Isso atacaria diretamente o custo
   de treino.
