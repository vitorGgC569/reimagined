# Piloto PT: educacional + sintéticos verificáveis + SFT completo

Receita `verified-pilot`, separada do experimento Synth reprovado. Nenhum registro
de `Polygl0t/gigaverbo-v2-synth`, peso antigo ou shard tokenizado antigo entra neste piloto.

## Orçamento e limites

- Base: 10M tokens reais empacotados, 95% GigaVerbo educacional e 5% tarefas locais.
- Continuação: 1M tokens de diálogos filtrados.
- SFT: 0,5M tokens únicos de resposta, duas épocas. Prompts não contam como alvos.
- Aproximadamente 12M tokens-alvo previstos; documentos/respostas são completos,
  por isso o preenchimento mínimo é 95%, sem cortes para fechar uma cota exata.
- Modelo nativo existente: Mamba-only, 16 camadas, dimensão 768, vocabulário 16384,
  contexto 512, pesos mestres FP32. Perfis baseline FP32, scan em blocos FP32 e
  scan em blocos BF16 para comparar no mesmo binário. Pesos inicializados do zero.
- É piloto de dados e integração. Não demonstra ganho de inteligência, superioridade
  ao estado da arte nem maturidade conversacional com esse orçamento.

## Verificação e isolamento

`ptbr_verified_tasks.py` gera cinco famílias: aritmética inteira, conservação de
estoque, conversões de unidades, ordenação e extração JSON. O verificador lê o
enunciado e a resposta publicados, não um gabarito auxiliar. Usa Decimal,
conservação, multiconjuntos/ordem e parsing JSON; nunca `eval` ou execução do texto.

O identificador canônico de problema define o holdout de 5%. Operandos comutados,
permutações de uma mesma lista e mudanças da ordem de chaves não atravessam a
divisão. Deduplicação lexical não elimina tarefas distintas apenas por usarem
o mesmo modelo de enunciado. Essa avaliação mede novas instâncias das mesmas
gramáticas; **não** mede generalização a famílias inéditas ou raciocínio aberto.

Cada exemplo tem prova, hash, índice de geração e semente. Antes de treinar,
todas as provas são recalculadas e confrontadas com payload/split/identidade.
Os hashes do gerador, verificador, receita e auditoria estão ligados ao pack.

## Dados externos

GigaVerbo base: revisão `b39dfa703102a20dc609ed6e7aaae22e8e3a233f`.
SFT: revisão `845713c9330519809c104fa904481c57ae948789`.
Educacional >=4, toxicidade <=1, política de fontes existente `commercial-strict`,
nenhuma família ultrapassa 60% dos tokens educacionais efetivamente empacotados.
Isso não constitui liberação jurídica de todas as páginas originais.

Candidatos educacionais já coletados no experimento v2 podem ser relidos em modo
somente leitura e novamente filtrados. O score de toxicidade desses candidatos
é atestado pelo ingest anterior, não reclassificado; a linhagem registra essa
limitação explicitamente. Os candidatos novos guardam também os scores brutos.

SFT/continuação: general 60%, math 15%, retrieval 15%, structured 10%, score >=4,
papéis alternados, resposta final completa, limite de contexto e filtros de
PII/boilerplate. Mesmo prompt tem a mesma fase e split. Dedup lexical entre fases
prefere holdout. Classificação de idioma cobre todo o documento em janelas,
não somente seu início. Fatos dos dados externos não são certificados.

## Operação Windows/HIP

```powershell
.\scripts\start_ptbr_verified_pilot.ps1 -Action run
```

Requer build `build-gm-hip`, TheRock em `C:/TheRock/build`, Python 3.12 com
datasets/tokenizers/numpy/pyarrow e langid 1.1.6 já instalado no diretório isolado
`artifacts/ptbr_edu_synth_20260920_v2/dependencies`.

Workspace v2: `artifacts/ptbr_verified_pilot_20260928`.
Logs `job-*.out.log`/`.err.log`, estado `job_status.json`, auditoria em
`corpus/quality_audit.json`, misturas em `packs/*/mixture_audit.json`, identidade
e checkpoints em `runs/main`. Checkpoint a cada 500 passos ou 10 minutos.
O launcher bloqueia outro processo PT ativo. `run` retoma o checkpoint compatível.
As leituras do Hub usam timeout de 60 segundos: a conexão local apresentou
respostas válidas em 16–26 segundos, acima do timeout padrão de 10 segundos.
Alterações da receita exigem outro workspace. Uma perda finita comprova execução,
não qualidade linguística. Os testes devem ocorrer após concluir a implementação.

## Correção da preparação e perfis de execução — 28/09/2026

A preparação v1 terminou a coleta, mas a avaliação educacional possuía somente
HPLT: o limite global foi preenchido antes da chegada de CrawlPT/Quati. O pack
parou com 28.334 tokens educacionais, abaixo de 47.500, por respeitar o teto de 60%.

A v2 recupera os candidatos do banco v1 em outra pasta, verifica receita, revisão,
hash de conteúdo e todas as provas sintéticas. Cada fonte recebe uma reserva de
60 mil tokens estimados para avaliação, escolhida por ordenação de hashes.
Todo holdout anterior permanece holdout. A deduplicação entre fases ocorre depois
da reserva e o tokenizer é treinado novamente, excluindo o novo holdout.

O caminho de recuperação evita novo download e nova classificação dos mesmos
dados. A alternativa para coleta nova limita a leitura a um shard remoto por vez
e antecipa a rejeição de fontes saturadas, que consumiu 247.412 linhas na v1.

`job_status.json` recebe passo, loss e velocidade a cada intervalo de log; o campo
`training_started` só passa a verdadeiro após um passo efetivamente concluído.
Os perfis são explícitos no launcher e na identidade nativa de cada checkpoint:

```powershell
.\scripts\start_ptbr_verified_pilot.ps1 -Action prepare
.\scripts\start_ptbr_verified_pilot.ps1 -Action train -Profile baseline -MaxTrainSteps 200 -RunName bench-fp32
.\scripts\start_ptbr_verified_pilot.ps1 -Action train -Profile chunked-fp32 -MaxTrainSteps 200 -RunName bench-chunked-fp32
.\scripts\start_ptbr_verified_pilot.ps1 -Action train -Profile chunked-bf16 -MaxTrainSteps 200 -RunName bench-chunked-bf16
.\scripts\start_ptbr_verified_pilot.ps1 -Action train -Profile chunked-bf16
```

As comparações devem rodar sequencialmente, usando o mesmo pack, seed e binário.
BF16 altera a precisão das GEMMs; gradientes e pesos mestres continuam FP32.
O ganho desta máquina deve ser medido nos logs atuais antes de ser afirmado.

## Validação anterior (v1)

Executada após a implementação: 44/44 testes Python, 58/58 testes da configuração
CPU. Na suíte HIP, 107/108 passaram; `test_gpu_parity_mamba_proper` excedeu a
tolerância de gradiente (`delta=0,001442`, limite `0,001295`) e passou depois em
10/10 repetições isoladas, sem alteração. Isso é uma intermitência observada,
não uma correção demonstrada. O caminho faithful utilizado no piloto e suas
variantes de paridade/checkpoint passaram na suíte. Logs preservados no workspace.

Binário HIP validado, SHA-256:
`3de2df4510db61d9cba7e7170b11609246bcfc02b14087c061c9837d06de1388`.

## Admissão SFT por tokens reais — 30/09/2026

A receita `ptbr-verified-pilot-v2.2` admite exemplos SFT contando os tokens da
resposta completa mais EOS com o tokenizer congelado. Prompt e resposta devem
caber integralmente no contexto. A reposição usa as mesmas fontes e revisões
fixadas, preserva os splits e executa novamente a deduplicação. A margem de
coleta é expressa na unidade correta; os mínimos finais dos packs permanecem.

Preparação concluída em `artifacts/ptbr_verified_pilot_20260930`, com 18 testes
Python passando e o comando `verify` concluído com código zero. O pack SFT de
treino contém 3.827 exemplos e 499.925 tokens de resposta: general 300.000,
math 74.934, retrieval 74.999 e structured 49.992. A avaliação contém exatamente
5.000 tokens, divididos em 3.000/750/750/500 nas mesmas categorias. Ambas as
auditorias registram zero prompts e respostas truncados.

Os packs base/continuação contêm respectivamente 9.999.832/999.545 tokens de
treino e 49.788/50.000 de avaliação. Identidade de packing:
`501a6128fde9c9f97d6c04745cde9264303714070552ffe106a067fcf07cb7d8`.
Evidência em `artifacts/maestri_integral_20260930/data-final-verify.log` e nas
auditorias `packs/*/mixture_audit.json`. Essa preparação usou o módulo nativo
existente somente para tokenização. Não certifica o novo runtime nem ganho de
qualidade: treinamento e ablação dependem da validação dos binários finais.
