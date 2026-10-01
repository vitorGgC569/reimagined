# Atividade dos gradientes esparsos — correção do contrato

Este lote corrige atividade host dos experts em todos os caminhos do Trainer.
Não implementa atividade/Adam inteiramente no dispositivo, não remove a
leitura tardia dos offsets MoE e não encerra o objetivo integral/release GPU.

## Lacuna e implementação

O armazenamento estável de `Parameter::grad` sobrevivia ao zeramento. O
Trainer usava `grad.size > 0` como atividade: um expert já selecionado podia
continuar na coorte Adam quando não recebia contribuição no grupo seguinte.
Isso não equivale à semântica de gradiente ausente. Os testes antigos de
expert vazio em um único backward não certificavam alternância entre steps.

- Experts Jamba e operadores grouped standalone registram explicitamente
  contribuição esparsa. Weight, bias, magnitude, parâmetros legados e LoQA
  pertencem ao mesmo contrato; parâmetros não treináveis seguem excluídos.
- `has_gradient()` distingue contribuição do grupo e buffer alocado. Um
  gradiente contribuído numericamente zero continua ativo. Não há inferência
  de atividade por norma, valores ou escolha de top-k no host.
- O reset ocorre ao abrir/descartar o grupo, inclusive no zeramento fundido.
  Não ocorre entre microbatches. Contribuições formam a união do grupo.
- Primeiro `add_grad` do grupo copia no buffer reutilizado; contribuições
  posteriores somam. Nenhum novo endereço é necessário na reativação.
- Finitude, escala, clipping, preflight, Adam fundido, determinístico flat/
  chunked, per-tensor, seleção de criticality e auditoria usam a atividade.
- Experts inativos não avançam versão, não sofrem Adam/weight decay e não
  decaem momentos. Experts nunca selecionados não criam momentos por tarefa.
- A regularização QAT global continua podendo contribuir para um expert que
  não recebeu tokens. Isso é uma contribuição real do objetivo, não atividade
  de tarefa: não foi silenciosamente convertida em regularização apenas ativa.
- Clones runtime copiam status além dos valores; snapshots portáveis continuam
  sem gradientes. O snapshot não consome os gradientes do modelo fonte.
- `Parameter::zero_grad()` passa a usar `gpu::current_stream()`, não stream 0.
- Python expõe `gradient_present` separadamente de tamanho de armazenamento;
  o probe não chama buffers retidos de gradientes ativos.

Parâmetros densos preservam a API legada de atribuição direta a `.grad`.
Para sparse registrado, escritas externas precisam publicar explicitamente
`mark_gradient_contribution()` após uma escrita válida. `add_grad` publica
automaticamente apenas depois de sucesso. A política atual é HOST.

## Identidade e compatibilidade

Modelos com parâmetros registrados incluem
`optimizer.sparse_gradient_policy=explicit_group_contribution_host_v1`.
Modelos sem experts não ganham esse campo. Sidecars v10 sem o campo não são
equivalentes: load rejeita a identidade antes de staging dos momentos.
Pesos para inferência não são tornados incompatíveis por essa correção.
A migração legada anterior, se explicitamente solicitada, não é prova de
continuação exata. Checkpoints antigos MoE não devem ser automaticamente
retomados como se a trajetória estivesse preservada.

O configurador Python requer a política para arquiteturas MoE, mesmo ao
solicitar compute legado. Um binário antigo não pode ignorar a correção.

## Validação e limitações

Artefatos: `artifacts/gpu_sparse_activity_20260930`.
Baseline anterior preservado SHA256
`551f3d7fadf009c693dae567cfeed2226f56d38b047ec133162ecf6a68737837`.
Seu diretório rocBLAS é uma junction para o build, não um pacote autocontido.

Após implementação, **CPU60/60** passou (54.38 s), **HIP121/121** passou
(147.54 s), e os **15 testes Python** de política passaram. Os dois builds
Release completos passaram. A primeira compilação do fixture tentou acessar uma API
privada do Trainer; o teste foi corrigido para o snapshot público sem alterar
a visibilidade da API. Falhas iniciais estão preservadas nos logs.
Logs finais: `cpu-full.log`, `hip-full.log`, `python-policy.log`.

HIP SHA256: `30e25d7157101c463a9f13fda957ca89ad62d32bfd594442f842c3f5f2c48b8c`.
CPU SHA256: `ebc8c110dfb1d798f8e8f7560e99e11c41b5b9ff1d9c0c47d4fcd3147d9cd7b0`.

Cobertura nova em `sparse_gradient_contract.h`:

- registro, gradiente zero versus ausente, buffers estáveis, rejeição sem
  mutação e cópia do status ativo/inativo;
- roteamento real em quatro experts top-1, embedding positivo e ramo Mamba
  zerado para controlar a seleção: ativo -> inativo -> reativação;
- pesos, versões e momentos de inativos conservados; ausência de estado dos
  experts nunca escolhidos; recorrência Adam com momentos preservados;
- contribuição zero realmente processada pelo Adam; união de dois
  microbatches com experts distintos; abort limpa atividade sem avançar step;
- CPU estados32/4 bits (matriz8192 elementos exercita quantização real);
- GPU quatro caminhos Adam: fundido, determinístico chunked, determinístico
  flat e per-tensor. Executados também no braço WMMA BF16 do CTest;
- snapshot público entre passos ativos/inativos; as campanhas já existentes
  de checkpoint/resume executam novamente sob a identidade corrigida.

## Reprodução física antes/depois

`physical-before-after-cpu.log` compara os dois binários HIP carregados em
processos separados, **executando CPU**, FP32 determinístico, seed9107,
D64/S33/E4/k1/H128, vocab67, LR0.0002, weight_decay0.03, QAT/aux OFF.
Embedding é constante positivo, parâmetros Mamba são zerados e a gate recebe
uma matriz com apenas a linha do expert selecionado igual a um.
Após selecionar expert0 no primeiro step e expert1 no segundo:

- Anterior: os seis parâmetros traináveis do expert0 tinham gradiente zero,
  mas versão2->3 e alteração de peso/magnitude/bias entre 0.00013387 e
  0.00013575. A correção não é apenas uma hipótese de análise estática.
- Atual: os seis conservam versão2, alteração exatamente0 e
  `gradient_present=false`, mantendo buffers alocados.

O guard Python com o **binário antigo real** recusou MoE legado antes de
qualquer sample: `native_verified=false`, política sparse ausente.
Evidência: `stale-native-rejected.json`/`.log` (exit1 esperado).
Probe atual RX7600, WMMA BF16, D128/S64/B1, cinco steps, retornou
`executed_finite`; serve como integração de identidade/binding/caminhos,
não benchmark de linguagem ou prova de convergência.

Não há medida de speedup neste lote. Trajetórias antigas que atualizavam
experts inativos não são baseline matematicamente equivalente. Permanecem
atividade GPU, coorte device e metadados estáveis sem registro CPU, promoção
por qualidade/performance, Mamba-3 e as demais frentes do inventário integral.

## Última recompilação deste lote, antes da acumulação por banco

A revisão de registro tardio corrigiu a adoção de um operador dense já
usado/zerado sem inferir contribuição de `grad.size`. Suítes finais repetidas:
CPU60/60 em18.56 s (`cpu-full-final.log`), HIP121/121 em146.85 s
(`hip-full-final.log`), builds em `cpu-build-final.log`/`hip-build-final.log`.
HIP SHA256 `5c088f5b372529f18dc44970808d2a78be63a19afc350d16f481ceeccc9d8e3d`;
CPU SHA256 `62bfa579f298c775d18cc805d579d35d1b5ca93487873c9f9fb55bba537dd8af`.
Os probes físicos/WMMA acima referem-se à compilação anterior explicitamente
identificada por SHA, não a essa recompilação. O lote seguinte de acumulação
por banco também corrige o clone dense zerado seguido de registro sparse;
seus resultados e binários são independentes em
`GPU_MOE_GRADIENT_COMMIT_2026-09-30.md`.
