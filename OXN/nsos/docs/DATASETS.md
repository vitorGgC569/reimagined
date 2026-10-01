# NSOS Datasets, Proveniência e Qualidade de Corpus

> Referenciado por `PRODUCT.md` como "Datasets and curriculum loaders".
> Este documento cobre (a) os carregadores de dados do runtime, (b) a
> proveniência e licenciamento do corpus PT-BR efetivamente preparado, e
> (c) o relatório quantificado de qualidade desse corpus.
>
> Toda medição abaixo foi produzida por `scripts/corpus_quality_report.py`
> contra o corpus real em disco. Nenhum número é estimado.

---

## 1. Carregadores de dados do runtime

| Superfície | Arquivo | Papel |
|---|---|---|
| `DataLoader` (C++) | `src/dataloader_v2.cpp` | Produtor/consumidor de minibatches sobre arquivo binário de floats, uma thread de background |
| `dataloader.h` (C++) | `include/dataloader.h` | Shim de compatibilidade — apenas `#include "dataloader_v2.h"`; sem lógica própria |
| `SmartLoader` (C++) | `src/smart_loader.cpp` | I/O assíncrono posicional com `std::future` real e barreira de drenagem |
| `U16TokenShard` (Python) | `scripts/train_ptbr_conversational.py` | Shard uint16 memory-mapped; janelas viram inteiros Python sob demanda |
| `SFTRecordShard` (Python) | idem | Registros SFT com offsets compactos; payload decodifica sob demanda |
| `CorpusStore` (Python) | idem | Corpus SQLite com deduplicação por fingerprint e cursor de retomada |

O caminho de dados **não é gargalo de treino**: instrumentação nativa mede
`preparation_ms` em 1,92 ms de um passo de 327 ms (0,6%). Ver
`docs/benchmarks/RX7600_PERF_CAMPAIGN_2026-08-06.md`.

---

## 2. Política de licenciamento na ingestão

`BASE_SOURCE_RULES` (`scripts/train_ptbr_conversational.py`) classifica cada
fonte antes de admiti-la. Sob `--license-policy commercial-strict` (padrão),
apenas `allow` entra; `conditional` é rejeitado.

| Decisão | Fontes | Motivo registrado |
|---|---|---|
| `deny` | Carolina, XL-Sum, Bactrian, BrWaC, Instruct-PTBR | CC BY-NC-SA / CC BY-NC / licença não declarada / termos derivados de Llama |
| `allow` | HPLT, CrawlPT, OSCAR, BlogSet, Quati, LegalPT, GPT4All, UltraChat | CC0 / Apache-2.0 / CC BY 4.0 / MIT conforme card do GigaVerbo-v2 |
| `conditional` | FineWeb, CulturaX, mC4, Common Crawl, Wikipedia | ODC-By + Common Crawl ToU; CC BY-SA com obrigação de atribuição |
| `unknown` | qualquer fonte sem regra | rejeitada por padrão (fail-closed) |

Fontes sem regra explícita são **rejeitadas**, não admitidas — o padrão é
fail-closed.

---

## 3. Proveniência efetiva do corpus `pilot`

Medido no corpus preparado em
`artifacts/ptbr_conversational/pilot/corpus/corpus.sqlite3`.

| Fase | Split | Documentos | Tokens estimados |
|---|---|---:|---:|
| base | train | 81.620 | 80.197.579 |
| base | eval | 380 | 443.298 |
| continuation | train | 27.057 | 12.002.347 |
| continuation | eval | 270 | 121.652 |
| sft | train | 25.100 | 4.201.540 |
| sft | eval | 236 | 38.773 |

### Fonte única — achado relevante

| Fonte | Licença aplicada | Documentos | Tokens | Share |
|---|---|---:|---:|---:|
| `HPLT/HPLT2.0_cleaned` | `allow` (CC0) | 82.000 | 80.640.877 | **100,00%** |

**A fase base é 100% HPLT 2.0 cleaned.** Nenhuma outra fonte permitida
(CrawlPT, OSCAR, BlogSet, Quati, LegalPT) contribuiu tokens. A política de
licença está correta e conservadora, mas o corpus resultante **não tem
diversidade de domínio**: é um único crawl web filtrado.

Implicação para o produto: um modelo treinado só nisso tende a herdar o
registro e os vieses de texto web genérico. Diversificar exige ou admitir
fontes `conditional` com a devida atribuição, ou buscar mais fontes `allow`.

---

## 4. Relatório quantificado de qualidade

Gerado por:

```bash
python scripts/corpus_quality_report.py --preset pilot --sample 12000 \
  --json artifacts/ptbr_conversational/pilot/corpus_quality_report.json
```

Amostra determinística de 12.000 documentos por fase (seed 20260806). O script
abre o SQLite em modo `immutable` — não pode alterar um corpus de treino.

### 4.1 Integridade estrutural — saudável

| Métrica | base | continuation | sft |
|---|---:|---:|---:|
| Documentos vazios | 0 | 0 | 0 |
| Curtos (<32 palavras) | 0,00% | 0,00% | 0,01% |
| Duplicatas exatas | 0,00% | 0,00% | 0,00% |
| **Quase-duplicatas** | **0,00%** | **1,10%** | **1,38%** |
| Palavras p10/p50/p90 | 192/466/1254 | 74/159/897 | — |
| Letras não-latinas p99 | 0,004 | 0,001 | — |

O corpus é genuinamente PT-BR (fração não-latina desprezível) e praticamente
livre de duplicação.

> **Nota metodológica.** A primeira versão deste relatório mediu 90,3% de
> quase-duplicatas no SFT. Era **artefato do detector**: todo registro SFT
> carrega o mesmo system prompt, e a regra original marcava duplicata ao
> compartilhar um único shingle. O detector agora exige que **mais de 50% dos
> shingles** de um documento já tenham sido vistos. Números acima são os
> corrigidos.

### 4.2 Vazamento train/eval — zero

| Fase | Sobreposição | Contaminação do eval |
|---|---:|---:|
| base | 0 | 0,0000% |
| continuation | 0 | 0,0000% |
| sft | 0 | 0,0000% |

O particionamento por hash de conteúdo (`_split_is_train`) cumpre o que
promete: nenhum documento de avaliação aparece no treino.

### 4.3 Distribuição de qualidade na ingestão

`edu_int_score` registrado por documento (fase base), filtro de admissão `≥3`:

| Score | Documentos |
|---:|---:|
| 3 | 65.418 |
| 4 | 16.491 |
| 5 | 91 |

Média 3,20. **O corpus é dominado pelo mínimo aceitável**: 79% dos documentos
têm exatamente o score de corte, e apenas 91 documentos (0,11%) atingem o
score máximo. Elevar o corte para `≥4` reduziria o corpus a ~20% do tamanho —
uma escolha real entre volume e qualidade que deve ser tomada com dados.

### 4.4 Boilerplate residual — o achado acionável

Padrões que o ingest **já filtra** deixam resíduo abaixo de 1% cada
(`clique aqui` 0,71%, `leia mais` 0,63%, `todos os direitos reservados` 0,47%).
O filtro do ingest opera por linha; o relatório mede por documento, então os
dois números não são diretamente comparáveis.

O problema real está no que o ingest **não filtra**:

| Padrão não coberto | base | continuation | sft |
|---|---:|---:|---:|
| **`url_residue`** | **7,32%** | 0,81% | 0,78% |
| **`email_residue`** | **4,20%** | 0,23% | 0,27% |
| `comment_count` | 1,43% | — | — |
| `copyright_line` | 1,23% | 0,31% | 0,49% |
| `boilerplate_ellipsis` | 0,68% | — | — |
| `social_cta` | 0,26% | — | — |
| `nav_breadcrumb` | 0,23% | — | — |
| **Total de docs afetados** | **13,45%** | 1,63% | 1,89% |

**Dois itens exigem decisão antes do lançamento:**

1. **`email_residue` em 4,20% da fase base é questão de privacidade, não de
   estilo.** São endereços de e-mail reais de páginas web dentro do corpus de
   treino. Um modelo treinado sobre isso pode memorizar e emitir PII. Para um
   lançamento oficial isso deve ser removido ou mascarado na ingestão.
2. **`url_residue` em 7,32%** ensina o modelo a emitir URLs — frequentemente
   inventadas na geração, o que é um vetor direto de alucinação verificável.

Ambos são correções de uma linha cada em `BOILERPLATE_PATTERNS`, mas exigem
**re-preparar o corpus** (o corpus atual foi construído antes desta análise).

---

## 5. Recomendações

| # | Ação | Custo | Motivo |
|---|---|---|---|
| 1 | Mascarar e-mails na ingestão (`[EMAIL]`) | 1 linha + re-prepare | PII em 4,20% da base |
| 2 | Remover ou mascarar URLs | 1 linha + re-prepare | Alucinação de links em 7,32% |
| 3 | Decidir corte de qualidade (`≥3` vs `≥4`) | decisão | 79% do corpus está no mínimo |
| 4 | Diversificar fontes além de HPLT | trabalho de dados | 100% do corpus é uma fonte só |
| 5 | Rodar este relatório como gate antes de cada treino longo | já pronto | evita queimar dias sobre corpus não inspecionado |

Os itens 1 e 2 devem preceder qualquer execução da receita `full`
(~6,5 dias de GPU): re-preparar custa horas, re-treinar custa dias.

---

## 6. Reprodução

```bash
# relatório completo, com JSON versionável
python scripts/corpus_quality_report.py --preset pilot --sample 12000 \
  --json artifacts/ptbr_conversational/pilot/corpus_quality_report.json

# corpus alternativo
python scripts/corpus_quality_report.py --corpus <caminho.sqlite3> --sample 20000
```

O script é somente-leitura por construção (`immutable=1` no URI do SQLite) e
determinístico dada a seed.
