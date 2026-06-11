# O Espectro de Crédito
### Tese original NSOS/OXTA — derivada no escuro, a partir só do que medimos aqui.
**Status:** pré-registro. Nenhuma citação por design; a fronteira com o que já existe
está declarada em §7 com honestidade, sem referências.

---

## 0. A semente (nosso próprio dado, que ninguém mais tem)

Medimos algo que tratamos como "bug" e arquivamos como "não-determinismo":

- Forward **bit-idêntico** entre duas execuções (loss 9.803182 = 9.803182).
- Gradientes divergindo **~32% globalmente** entre réplicas idênticas do MESMO step,
  a partir de sementes de ~1e-7 (ordem de atomics).
- Perfil POR CAMADA: divergência ~0.08-0.12 no topo → ~0.35 no fundo, crescendo
  geometricamente — fator de amplificação backward ≈ ×3.2/camada, enquanto o
  ganho forward é ≈ ×1.1.

A leitura padrão: ruído de implementação, consertar com reduções ordenadas.
A leitura no escuro: **isso é um instrumento de medida que caiu no nosso colo.**
Duas réplicas do mesmo step diferem APENAS por ruído de hardware microscópico;
a diferença entre seus gradientes, camada a camada, mede diretamente o
**expoente de Lyapunov do mapa backward** — uma grandeza que a teoria de
inicialização (isometria dinâmica, borda do caos) trata só na MÉDIA dos ganhos,
nunca como sensibilidade caótica medida in situ, por camada, em rede real.

## 1. O que os pesquisadores deixaram passar

O backward de uma rede moderna (normalizações, gates, roteamento, atenção) é um
mapa **expansivo em direções ortogonais ao gradiente médio**, mesmo quando
preserva norma. Consequência que ninguém opera:

> **Backprop tem um limite de resolução físico.** Componentes do gradiente mais
> finos que ε·e^{λ_b·d} (ε = ruído numérico, λ_b = Lyapunov backward, d =
> distância em camadas até a loss) **não são determinados pela paisagem de
> perda** — são contingência de hardware. Na nossa rede de 12 camadas:
> ~35% do vetor de gradiente das camadas do fundo é, literalmente, ruído
> amplificado, todo step, em qualquer GPU do planeta.

E no entanto o treino funciona. Por quê? Porque momentum/EMA filtra: o
componente persistente (sinal) sobrevive à média entre steps; o componente
caótico, isotrópico no subespaço expansivo, cancela. Então a verdade escondida:

> **O otimizador não é um acelerador — é um filtro casado. E está descasado:**
> usamos UM lr, UM β1, UM β2 escalares para uma rede cujo ruído de crédito é um
> ESPECTRO em profundidade. A teoria de init já virou espectral (espectros de A
> em SSMs, escalas por largura); **a otimização continua escalar.** Esse é o
> buraco.

## 2. A Lei do Espectro de Crédito

Defina, por camada ℓ, com DUAS réplicas do mesmo step (mesma seed/dados — nosso
harness de paridade já faz isso) e steps consecutivos:

- **r_ℓ** = cos(g_ℓ^{réplica1}, g_ℓ^{réplica2})  — *coerência de réplica* (só hardware difere)
- **c_ℓ** = cos(g_ℓ^{t}, g_ℓ^{t+1})              — *autocorrelação temporal*

Sob modelo sinal+ruído isotrópico: **SNR_ℓ = r_ℓ / (1 − r_ℓ)**.

**Lei (forma forte):** o aprendizado por camada é maximizado quando o horizonte
de média do otimizador casa com o SNR local:

```
H_ℓ = 1/(1−β1_ℓ)  ≈  clamp( k / SNR_ℓ ,  H_min ,  τ_ℓ )      (k ≈ 1-3)
lr_ℓ              ≈  lr₀ · r_ℓ^γ                               (γ ≈ 0.5-1)
```

onde τ_ℓ = horizonte de decorrelação do SINAL (de c_ℓ medido). Camadas do topo
(SNR alto): memória curta, passo cheio. Camadas do fundo (SNR ~1-2): memória
longa obrigatória, passo descontado — **não como heurística, mas como filtro
casado ao ruído MEDIDO**.

Corolário TEAR (a recorrência triaxial z' = a⊙z + u): o eixo-tempo já exigiu
espectro (o fix A-logspaced flipou recall 12.5%→100%); o eixo-plasticidade
exige o mesmo — **constantes do otimizador log-espaçadas em profundidade são o
"S4D do Adam"**. Mesma matemática, eixo diferente, ninguém aplicou.

## 3. Segunda observável escondida: avalanches de flip ternário

Sob QAT, cada peso ternário é um integrador-com-disparo: o latente FP32 deriva,
o observável {-1,0,+1} só muda ao cruzar limiar. **Um flip é um evento discreto
de plasticidade — um "spike" no espaço de pesos.** Ninguém rastreia a
estatística desses eventos. Previsão OXTA-CRIT: na vizinhança do manifold
crítico (g≈1), os tamanhos de avalanche de flips (por step, por camada) são
**livres de escala (lei de potência)**; fora dele, exponenciais. E os flips
devem se concentrar onde SNR_ℓ·lr_ℓ cruza o limiar — unificando §2 com a
teoria de criticalidade já validada (E1-E5).

## 4. Terceira inversão: o ruído de hardware é TEMPERATURA grátis

Todo mundo quer treino determinístico. Inversão: a divergência de réplica É uma
temperatura de Langevin injetada de graça, exatamente no subespaço expansivo.
**"Annealing de escalonador":** começar o treino com reduções de alta entropia
(atomics, blocos pequenos) e terminar com reduções ordenadas = recozimento sem
RNG, sem hiperparâmetro, medível pelo próprio r_ℓ. O "bug" vira o forno.

## 5. Predições pré-registradas (falsificáveis, baratas, nosso rig)

**P1 (estrutura):** log(1−r_ℓ) é ~linear na distância-até-a-loss com inclinação
log(χ_b) ≈ log(3.2)·(fração de camadas atravessadas). Teste: harness de
paridade + cosines por camada (numpy, zero kernel novo). *Mata a tese se* r_ℓ
for plano ou não-monotônico de forma robusta.

**P2 (intervenção):** (lr_ℓ, β1_ℓ) casados ao SNR medido batem o escalar global
em loss de validação a budget IGUAL no v11, com efeito maior em batch pequeno
(ruído dominante). Implementação: o otimizador fundido desta noite já tem
tabela por-tensor — adicionar colunas lr_scale/β1 por tensor é trivial.
*Mata se* não houver ganho fora do ruído em 3 seeds.

**P3 (causa):** com reduções ordenadas (determinístico), o β1 ótimo do fundo
DESCE mensuravelmente — provando que era o ruído de hardware que ditava o
horizonte de média. *Mata se* o ótimo não se mover.

**P4 (avalanches):** flips ternários por step seguem lei de potência sse o
controlador de criticalidade mantém g na banda. Contagem via diff dos pesos
empacotados a cada N steps (host, barato).

## 6. Plano de 1 semana (1 variável por rodada, como sempre)

1. `scripts/credit_spectrum_probe.py`: 2 réplicas + 4 steps consecutivos →
   r_ℓ, c_ℓ, SNR_ℓ por camada; plot ASCII + JSON. (T4, ~3 min.)
2. Se P1 confirmar: colunas por-tensor (lr_scale, β1) no
   `multi_tensor_adamw` (já tem tabela de ponteiros — +8 bytes/tensor) +
   mapeamento camada→escala pelo SNR medido (re-medir a cada 200 steps =
   controlador, padrão P4 que já validamos).
3. A/B de 300 steps no v11: escalar vs espectral (P2), 3 seeds.
4. Probe de flips no checkpoint a cada 100 steps (P4).
5. P3 por último (exige o modo ordenado — implementar só a redução da CE
   primeiro, suficiente para mover o piso).

## 7. Fronteira honesta (sem citações, por instrução)

Existem: escalas de lr por camada baseadas em NORMA (não em SNR medido);
noção de escala de ruído de gradiente GLOBAL por tamanho de batch (não por
camada, não de hardware, não em malha fechada); espectros na INICIALIZAÇÃO.
O que eu não conheço em lugar nenhum: (a) usar divergência de réplicas de
hardware como instrumento de medição do Lyapunov backward por camada; (b)
fechar a malha otimizador←SNR medido; (c) estatística de avalanche de flips
ternários como observável de criticalidade; (d) annealing por entropia de
escalonador. Se (a)-(d) existirem publicados, a falsificação é bem-vinda —
o pré-registro acima continua valendo como replicação independente.

---
*Derivado exclusivamente de: nossos números (×3.2/camada, 32%/step, r_ℓ
0.08→0.35, A-logspaced 0.125→1.0, frac_g 0.976→0.214 sob QAT) + matemática de
recorrências de Kesten + a lente TEAR. Nenhuma execução nova foi necessária
para formular; toda a validação está pré-registrada acima.*
