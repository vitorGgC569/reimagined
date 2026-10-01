# NSOS Project Boundary

> Referenciado por `INCUBATION.md` como o documento que codifica a fronteira
> do produto. Define o que é produto suportado, o que é incubação e quais
> regras governam a passagem entre os dois estados.
>
> Autoridade: `PRODUCT.md` decide **capacidade**; este documento decide
> **pertencimento**. Quando divergirem sobre o que existe, `PRODUCT.md` vence.

---

## 1. Dentro da fronteira (produto suportado)

| Árvore | Papel |
|---|---|
| `OXN/nsos/include`, `OXN/nsos/src` | Runtime C++20 (`nsos_core`) e superfícies derivadas |
| `OXN/nsos/CMakeLists.txt` | Única fonte de verdade sobre o que compila |
| `OXN/nsos/tests` | Suíte CTest registrada; a configuração falha se um `test_*.cpp` ficar sem classificação |
| `OXN/nsos/scripts` | Pipeline de currículo, treino, gates e benchmarks |
| `OXN/nsos/eval` | Harness de avaliação externa |
| `OXN/nsos/python` | Pacotes wrapper sobre `nsos_ext` |
| `OXN/nsos/docs` | Documentação do produto |
| `modules/oxtamem` | Motor de memória em Rust, integrado por FFI dinâmica |

Alvos de release: `nsos_core`, `nsos_ext`, `nsos_cli`, `nsos_api_server`.

## 2. Fora da fronteira (incubação)

Inventariadas em `INCUBATION.md`. Nenhuma é compilada por
`OXN/nsos/CMakeLists.txt` nem referenciada por `PRODUCT.md`.

`KernelOpen/`, `CHRASS/` (árvore standalone, distinta da camada
`chrass_layer_v2` do runtime), `CART/`, `OXB/`, `hardware/`, `bindings/`
(raiz), `include/pantheon/` + `src/pantheon/`, `research/`, `experimental/`,
`tools/`, e as árvores paralelas de raiz `src/`, `include/`, `tests/`,
`scripts/`, `benchmarks/`.

## 3. Fora da fronteira (congelado)

`legacy/` — append-only. Nada em `legacy/` está em caminho de execução.
Mover para `legacy/` é reversível: o histórico permanece em git.

---

## 4. Regras invioláveis

1. **Sem back-reference do produto.** Nenhum arquivo sob `OXN/nsos/` ou
   `modules/oxtamem/` pode incluir ou importar de uma árvore de incubação.
2. **O CMake é a verdade.** Se não está em `NSOS_CORE_SOURCES` ou em um
   `add_executable`/`add_test`, não é produto — independentemente de onde o
   arquivo esteja.
3. **Sem promoção silenciosa.** Promover exige um PR único que atualize
   `PROJECT_BOUNDARY.json`, `PRODUCT.md` e `INCUBATION.md` juntos.
4. **Código de demonstração é fail-closed.** Headers que narram
   comportamento em vez de implementá-lo exigem
   `NSOS_ALLOW_DEMONSTRATION_HEADERS`; incluí-los sem opt-in é erro de build.
   Aplica-se hoje a `predictive_failure.h`, `kernel_reflection.h`,
   `entropy_manager.h` e `nsos_boot.h`.
5. **Toda alegação de desempenho precisa de artefato reproduzível** commitado
   (script + relatório), não de número em README.

---

## 5. Critérios de elegibilidade a produto

Uma árvore só é promovível quando **todos** forem verdadeiros:

1. Razão de produto declarada e responsável nomeado.
2. Build determinístico a partir de checkout limpo.
3. Testes unitários e de integração registrados no runner do produto.
4. Caminhos de runtime/serviço com autenticação, limites de recurso e testes
   de modo de falha, quando aplicável.
5. Alegações de desempenho reproduzíveis por script commitado.
6. Documentação atualizada em `README.md`, `PRODUCT.md`, `INCUBATION.md` e
   neste arquivo.
7. Verificação de fronteira verde (`scripts/check_project_boundary.py`).
8. Pelo menos uma execução de scorecard no degrau `pilot` sem regressão nas
   métricas de `SCORECARD.md`.

Se uma árvore não puder atingir o critério 8 dentro do próximo release
planejado, a resposta correta é `not now` ou `archive` — nunca "promover
assim mesmo".

---

## 6. Estados possíveis

| Estado | Significado | Efeito |
|---|---|---|
| `product` | Dentro da fronteira | Coberto por gates de release |
| `incubating` | Valor real, trabalho nomeado pendente | Falhas não bloqueiam release |
| `archive` | Real, mas sem destino | Move para `legacy/` |
| `extract` | Não é material de produto de software | Sai para repositório próprio |
| `delete` | Sem valor e sem histórico a preservar | Removido |

A atribuição por árvore vive em `INCUBATION.md`.
