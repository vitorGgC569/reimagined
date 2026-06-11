# Economia do Flip — pós-fogo (rodada adversarial 3)
**Proposta original:** o flip ternário como evento atômico → (1) updates de
frota por delta-de-flips, (2) flip-rate como telemetria, (3) replay dirigido
por flip revertido.  **Veredito do árbitro:** 1 rebaixado, 2 condicional,
3 morto.  Registro honesto + o que fazer.

## Mortos e rebaixados (colisões nomeadas)
- **Claim 3 (replay por flip revertido): MORTO.** É MIR (Aljundi 2019)
  reinventado sobre atribuição causal insustentável (flip = última gota de
  centenas de updates; SI de Zenke formaliza o certo); janela M contraditória;
  falso negativo estrutural (interferência via tensores FP/latente sem flip).
  Kill-criterion aceito: zero esforço nisso antes do gap nº1 (generalização).
- **Claim 1 (delta-de-flips): REBAIXADO a nota de engenharia.** Baseline
  honesto é a imagem ternária packed (~10MB @40M), não os 395MB FP32; e o
  estado funcional inclui tensores FP que mudam todo step (embedding, router,
  normas, flat_beta, SSM) ⇒ piso de MBs sem um plano de congelar/quantizar
  esses tensores.  Colisões: Delta-DNN, LC-Checkpoint, BitDelta, DeltaZip,
  Konečný 2016.  Vira relevante em ~1B params com inventário FP resolvido.
- **Claim 2 (flip-rate telemetria): CONDICIONAL.** Colide com Bop (flip-rate
  como análogo de LR) e Nagel 2022 (oscilação de níveis em QAT → dampening/
  freezing).  Só vale se ANTECIPAR o que grad-norm e update-to-weight ratio
  (grátis) não antecipam — pré-registrado como comparação de 3 seeds; confounder
  de oscilação-de-fronteira exige medir histerese/idade de flip primeiro.

## O que sobrevive e AJUDA JÁ (custo zero de quota)
**EXP-FLIP-0 (CPU runtime, checkpoints existentes no Drive):** diff
step100×step200×step300 do run atual com o protocolo do árbitro:
1. quantizar os 3 com o MESMO procedimento do repack;
2. taxa de flips por camada por 100 steps; fração de REVERSÕES (oscilação) vs
   flips persistentes — isto é o pré-requisito F4 e alimenta DIRETO o P5
   (avalanches) e o P6b (lei η·σ_g/Δ) da Tessitura pós-fogo;
3. tamanho entropy-coded do delta INCLUINDO tensores FP, vs baseline
   bsdiff/xdelta da imagem packed (aritmética honesta do Claim 1);
4. de brinde: primeira medição real de "idade de flip" (histerese) do projeto.

## Lição acumulada do método (3 rodadas)
Ideias grandiosas → fogo adversarial → sobra pequeno, afiado e mensurável.
Saldo das 3 rodadas: S1-S5 (Tessitura) + EXP-FLIP-0 + telemetria condicional —
um programa experimental inteiro construído por debate entre instâncias, cada
experimento com controles desenhados por quem tentou matá-lo.
