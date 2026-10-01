# Incubation Inventory — decisões da Fase 7

> **Propósito:** tornar explícita a resposta para cada diretório não-produto
> deste monorepo. Se uma pasta não é `OXN/nsos`, `modules/oxtamem`, `legacy/`
> ou suporte de release, ela precisa aparecer aqui com uma decisão concreta.
>
> **Status:** Fase 7 executada em 2026-08-07. Os campos de decisão deixaram de
> ser placeholders. As evidências abaixo vêm de leitura estática do código e
> foram verificadas em primeira mão nos pontos mais graves.
>
> **Regra:** nenhuma árvore de incubação pode ser referenciada por
> `PRODUCT.md` nem entrar em artefato de release. Falhas aqui não bloqueiam os
> gates de produto, exceto durante uma promoção.

---

## Como uma árvore sai da incubação

Critérios em `docs/PROJECT_BOUNDARY.md` §5. Resumindo: razão de produto e dono
nomeados, build determinístico, testes no runner do produto, limites de
recurso onde aplicável, alegações de desempenho reproduzíveis por script
commitado, documentação atualizada, verificação de fronteira verde, e uma
execução de scorecard `pilot` sem regressão.

Se uma árvore não puder atingir o último critério dentro do próximo release
planejado, a resposta certa é `not now` ou `archive` — nunca "promover assim
mesmo".

---

## Achados transversais

**1. O `CMakeLists.txt` da raiz é infraestrutura morta.** Ele compila
`KernelOpen` + `bindings/` + `src/pantheon/` num módulo Python `pantheon`.
Nenhum Dockerfile e nenhum job de CI o invoca — todos configuram
`-S OXN/nsos`.

**2. A CI toca a incubação apenas com verificação de sintaxe, e não bloqueia.**
O job `incubation` roda `python -m compileall` sobre
`benchmarks scripts tests KernelOpen CHRASS CART OXB tools` com
`continue-on-error: true`. Não compila C++, não roda teste, não executa
benchmark. `research/`, `experimental/`, `bindings/` e `hardware/` não são
cobertos nem por isso.

**3. A regra de "sem back-reference" vale de fato, mas não é verificada.**
Grep de `OXN/nsos/src`, `include` e `CMakeLists.txt` por `pantheon/`,
`KernelOpen`, `CART/`, `OXB/`, `CHRASS/`: zero ocorrências. Porém
`scripts/check_project_boundary.py` valida apenas a estrutura do
`PROJECT_BOUNDARY.json` e a existência dos caminhos — **não** há checagem de
imports produto→incubação. A regra é honra, não gate.

**4. Duplicação massiva.** `KernelOpen/src/graph/{kimera,chrass,chrass_tsp,
rierass_core}.cpp` são **byte-idênticos** aos equivalentes em `CHRASS/`, e
nenhum está no CMake do KernelOpen. `KernelOpen/src/aion/*` é um fork
desatualizado de `OXB/aion_core_cpp/src/*` — o `BitPacking.cpp` do OXB recebeu
a correção de portabilidade MSVC, o do KernelOpen não.

**5. Artefatos de execução existem em apenas dois lugares:**
`research/lateral_inhibition/` (7 arquivos) e `CHRASS/` (2). `KernelOpen`,
`CART`, `OXB` e `hardware` têm **zero** resultados commitados — toda alegação
de desempenho nessas árvores existe só como prosa em README.

---

## Inventário e decisões

### `KernelOpen/` — **`extract` (repositório próprio), com remoção de alegações**

5.619 LOC. Último commit 2026-05-26.

**O que é real:** o escalonador em `src/host/uhk_runtime.h` (fila lock-free
com ring buffer e caminho CPU funcional) e `src/common/stream_k.h`
(decomposição Stream-K), ambos exercitados por `tests/unit_tests.cpp`.

**O que não é:** os módulos que dão nome à árvore. O simulador quântico é
fixo em 9 qubits sem decodificador; o "photonic engine" é multiplicação
matriz-vetor complexa; `src/ghost/holographic_memory.h:27` regenera a matriz
de projeção com `rand()` sem seed **a cada chamada**, de modo que codificar a
mesma entrada duas vezes produz vetores diferentes — o único teste que o
exercita recupera o mesmo objeto que acabou de armazenar e por isso nunca
detecta o defeito.

**Alegação de 4200 TOPS / 1,1 µs — verificada em primeira mão e refutada:**

- `src/device/persistent_kernel.cu:273` —
  `control_ptr->throughput = 4000.0f; // Hardcoded theoretical peak for now`,
  com o cálculo real comentado na linha acima.
- `frontend/app.py:55-56` —
  `kernel_state["throughput"] = 4000.0 + random.uniform(-100, 100)` e
  `kernel_state["latency"] = 1.1 + random.uniform(-0.1, 0.1)`.

O painel que exibe "4200 TOPS / 1,1 µs" gera os números com um RNG. A epígrafe
do próprio README é *"Nada absolutamente nada deve ser Simulação."*

**Decisão:** extrair `stream_k.h` e `uhk_runtime.h` para o produto **se** o
escalonador for desejado; apagar `src/graph/` (duplicata byte-idêntica de
CHRASS) e `src/aion/` (fork pior do OXB); mover o restante para repositório
pessoal **com as alegações de TOPS/latência/BCI removidas**. Não publicar sob
o nome do produto. Risco reputacional adicional: `docs/ARTIGO_TECNICO.md:31`
afirma capacidade de "controle mental de interfaces" sustentada por um
detector de limiar de 38 linhas sobre placa sintética.

**Revisão:** não aplicável após extração.

---

### `CHRASS/` (raiz — solver SSSP Kimera) — **`keep incubating`, condicionado**

7.754 LOC. Último commit 2026-05-01. Não confundir com
`OXN/nsos/src/chrass_layer_v2.cpp`, que é a camada do runtime e não tem
relação com esta árvore.

**O que é real, e é bom:** `kimera.cpp` é HPC sério — difusão de calor por
Chebyshev com OpenMP, reordenação WDD para localidade de cache, heap radix
monotônico de 64 buckets e relaxação de arestas em AVX2 escrita à mão
(`_mm256_i32gather_epi64`, `_mm256_cmpgt_epi64`) com cauda escalar correta.
É a peça de engenharia bruta mais impressionante do conjunto de incubação.

**A metodologia de benchmark também é acima da média:** `ultimate_benchmark.py`
clona e compila um baseline de terceiros real (`bmssp`) em vez de um
espantalho.

**Mas o artefato commitado contradiz o README — verificado em primeira mão.**
`CHRASS/output.txt`, único registro de execução:

```
Nodes: 1000000 | Mode: GRID
Total Time: 373.044 ms
Step3(Core): 0.014612 ms
Relaxations: 57
```

`Relaxations` incrementa uma vez por nó retirado do heap. **57 nós de
1.000.000** — o solver terminou após tocar 0,0057% do grafo, isto é, não
resolveu a instância. Os 0,0146 ms do "core" são o tempo de não resolver.
O README anuncia "SSSP Grid (1M) | 90 ms".

**As alegações RSA-2048/4096 e Shannon 10^120 medem outra coisa.**
`kimera_rsa4096.cpp:206` constrói o grafo com `generate_line(20000)` — um
**grafo caminho**, a topologia SSSP mais fácil possível — com peso constante
`2^4080` em toda aresta. O "resultado RSA-4096 em 65,87 ms" é o custo de
20.000 somas sequenciais de inteiros de 4096 bits numa lista ligada. Não diz
nada sobre RSA, fatoração ou segurança.

**Decisão:** manter em incubação, condicionado a três trabalhos nomeados:
(1) **oráculo de correção** — diff contra Dijkstra de referência em todos os
cenários, ligado a um runner; o artefato atual indica resposta errada no
próprio benchmark-bandeira; (2) **renomear as variantes de alegação
extraordinária** (`kimera_rsa*`, `kimera_shannon`, `kimera_poincare`,
`chrass_riemann`, `chrass_navier`, `god_algorithm`) para
`bench_wide_int_sssp_*` e corrigir o README; (3) **vendorizar ou fixar
`bmssp`** e revisar sua licença. Se (1) não produzir distâncias corretas em
esforço razoável, rebaixar para `archive`.

**Revisão:** próximo ciclo de release, ou imediatamente se houver intenção de
publicar.

---

### `CART/` — **`keep incubating`, a um sprint de `promote`**

3.148 LOC. Último commit 2026-05-26.

**Código real e competente:** 14 algoritmos PEFT com headers e implementações
correspondentes. A extensão LibTorch implementa DoRA corretamente
(`W' = m·(V/‖V‖)` com `V = W₀ + BA`), depois escala por IA³, com redimensiona-
mento dinâmico de rank que preserva pesos aprendidos por slicing. Nenhum stub
no caminho PyTorch.

**A alegação de 113× é insustentável como escrita** — e a própria árvore
admite: `VALIDATION_REPORT.md:91` diz *"Claim do README de '113× speedup' não
foi reproduzido aqui."* O script de verificação citado pelo README mede apenas
o lado PyTorch; não há baseline C++ nele. A comparação, quando feita, é contra
o motor escalar deliberadamente não otimizado dos próprios autores — "113×
mais rápido que nosso baseline ingênuo" não é alegação sobre nada.

**O que é sustentado:** 21 testes passando com números específicos e comandos
de reprodução, documentados honestamente.

**Decisão:** manter em incubação. Trabalho nomeado para promover: (1) apagar
ou provar o 113× com um script único que meça os dois backends na mesma
máquina e grave JSON; (2) adicionar `enable_testing()` + `add_test()` ao
CMake e um job de CI que rode `cart_validation` — cerca de uma hora de
trabalho, converte a árvore de "confie no relatório" para "a CI prova";
(3) implementar ou remover `IA3Layer::enablePureScalingMode`, declarado e
nunca implementado.

**Revisão:** próximo ciclo de release.

---

### `OXB/` — **`keep incubating` → candidata natural a `promote`**

3.968 LOC. Último commit 2026-05-26.

**A árvore que deve se formar.** Quatro algoritmos corretos e não triviais
(RMI learned index, bit-packing de N bits arbitrário, curva de Hilbert,
serializador OX3), com camada de portabilidade que funciona de fato: um
registrador de deslocamento de 128 bits escrito à mão e guardado por
`#if !defined(__SIZEOF_INT128__)` para compilar tanto em MSVC quanto em
GCC/Clang.

**Honestidade exemplar:** o próprio `VALIDATION_REPORT.md` mede as alegações
do MANIFESTO e publica a derrota — bit-packing 1,13 B ops/s alegado contra
406 M medido (2,8× otimista); RMI predict 533 M contra 302 M (1,8×); Hilbert
202 M contra 27 M (7,4×). Também registra que o AVX2 batch predict empata com
o escalar porque o compilador já vetoriza. E documenta que `pack_avx512()` é
placeholder que chama `pack_scalar()`.

**Cobertura de teste é a melhor do conjunto:** 16 testes C++ cobrindo
correção, casos de borda, determinismo e bijetividade do mapeamento de
Hilbert, mais 4 suítes pytest.

**Decisão:** manter em incubação com caminho curto para promoção. Tudo que
falta é procedimental: ligar `aion_validation` ao CTest e à CI; reescrever os
números do MANIFESTO com hardware/compilador anexados (ou apagá-los — os 406M
/302M auditados se sustentam sozinhos); apagar ou implementar `pack_avx512`;
implementar `WindowsSmartLoader` ou documentar explicitamente que é Linux-only.
**Não promover antes de a CI existir** — a evidência atual é um relatório
humano confiável, não um gate reproduzível.

**Revisão:** próximo sprint.

---

### `research/` — **`keep incubating` permanente; é a régua de qualidade**

5.657 LOC. Último commit 2026-06-10 — a árvore mais recente do conjunto.

**A metodologia mais rigorosa do repositório.** O README abre **refutando o
próprio resultado anterior do autor**: o POC original tinha baseline
inicializado em std=0,02 e depois arredondado para ternário, colapsando a rede
a zero na inicialização, enquanto a variante "Oxta" tinha um bypass FP32
escondido que fazia 100% do trabalho. O harness existe para impedir essa
classe de erro. Os princípios declarados incluem uma variável por vez, sem
bypass escondido, implementações de referência reais vindas do PyPI em vez de
espantalhos, e "honestidade acima de advocacia".

`speculative_claims.md` enumera nove alegações que o harness deliberadamente
**não** testa, cada uma com o bloqueador concreto — incluindo sinalizar que
"Hypertoken ECC não é termo padrão em nenhum paper".

`lateral_inhibition/` é a única árvore com **artefatos brutos de execução
commitados**, e reporta as derrotas junto com as vitórias.

**Decisão:** manter em incubação permanentemente, como harness de pesquisa ao
lado do produto — o próprio README argumenta corretamente que não usar o motor
C++ do NSOS evita contaminar medições de performance com código de pesquisa.
Trabalho nomeado: (1) rodar `run_all.py --full` uma vez e commitar
`results/<timestamp>/*.json` — o harness nunca foi executado até o fim, então
as 12 alegações seguem sem evidência registrada; (2) incluir `research` na
lista de `compileall` da CI, de onde está inexplicavelmente ausente;
(3) substituir os binários MNIST vendorizados por um passo de download.

**Revisão:** contínua.

---

### `experimental/` — **AÇÃO IMEDIATA: versionar antes de qualquer decisão**

4.505 LOC em 16 arquivos Python. **Verificado: `git ls-files experimental`
retorna 0 — a árvore inteira está fora do controle de versão.**
`.gitignore` cobre `*.bin`, mas os fontes Python simplesmente nunca foram
adicionados. Mtimes de 2026-07-27 fazem dela a árvore mais recentemente
trabalhada, e nada disso está em git.

`benchmark_nsos_honesto.py` tem **1.743 linhas** e é um harness de avaliação
do **produto**, não de incubação: recall associativo QAR sobre vocabulários
disjuntos, cópia seletiva, generalização OOD por comprimento, perplexidade com
teacher forcing, exact match, multi-seed com média/desvio/IC95, ablações sobre
OxtaMem/QAT, e separação train/val/test por hash. É mais completo que o gate
do próprio produto.

**Decisão:** (1) **commitar os 16 arquivos `.py` agora** — estão untracked,
não ignorados, a um `git add` de estarem seguros — mantendo `*.bin`, `*.log`,
`cache/` e `__pycache__/` fora; (2) decidir o destino: o harness honesto
provavelmente pertence a `OXN/nsos/scripts/` ou como irmão de
`research/theses_validation/`, não a uma pasta sem nome; (3) declarar
`experimental` e `research` em `PROJECT_BOUNDARY.json`, onde hoje não constam
— ambos caem na política padrão por omissão, não por declaração.

**Revisão:** imediata.

---

### `hardware/` — **`extract` (repositório de hardware) ou `delete`**

216 LOC. Último commit 2026-05-01.

O MAC ternário está correto, mas o header anuncia `ARRAY_SIZE = 64 // Systolic
Array Dimension (64x64)` enquanto a implementação é um banco 1-D de
acumuladores independentes **sem movimento de dados entre elementos** — é uma
linha de MAC, não um array sistólico. O controlador de memória é mock
explícito (`host_data_out <= 16'hBEEF`). As portas usam sintaxe SystemVerilog
(arrays não-empacotados) num arquivo `.v`. A alegação "~1000× vs FP32" não tem
relatório de síntese nem estimativa de potência. Não há simulador Verilog em
nenhum lugar do monorepo.

**Decisão:** extrair para um repositório de hardware com toolchain RTL e CI
reais, ou remover — a ideia do MAC ternário são três linhas de `case` e é
recuperável por `git log`.

---

### `bindings/` (raiz) e `include/pantheon/` + `src/pantheon/` — **`archive to legacy/`**

51 LOC e 1.444 LOC respectivamente. Ambos parados desde o snapshot inicial
(2026-05-01).

O Pantheon tem 22 headers e 7 `.cpp`, dos quais **6 são placeholders vazios**
que existem apenas para satisfazer a lista de fontes do CMake — um deles diz
isso literalmente: *"we create the cpp file to satisfy CMake source list."*
O sétimo é despacho puro. Os headers contêm código funcional, mas cada um tem
~25 linhas e nenhum é carregado por qualquer outro arquivo do monorepo.

`bindings/python_bindings.cpp` é um binding correto sobre esse motor, superado
por `OXN/nsos/src/bindings.cpp`.

**Decisão:** arquivar `include/pantheon/`, `src/pantheon/`, `bindings/` e o
`CMakeLists.txt` da raiz numa única mudança. Manter `artigopantheon.md` na
raiz, conforme decisão anterior. Se alguma perda específica for desejada
depois, portar aquele arquivo sob demanda é mais barato que manter a taxonomia.

---

### `tools/`, `scripts/` (raiz), `benchmarks/` (raiz), `tests/` (raiz) — **triagem**

- **`tools/`** (276 LOC) — contém a única costura produto↔incubação:
  `pack_dataset.cpp` inclui tanto `OXN/nsos/include/tokenizer.h` quanto
  `OXB/.../OX3Serializer.h`. Direção incubação→produto, então não viola a
  regra. **`archive`**.
- **`scripts/` (raiz)** — `check_project_boundary.py` é suporte de release e
  roda na CI: **fica**. O restante (`train_*.py`, scripts de build,
  `data_pipeline/`, `distillation/`): **`archive`**.
- **`benchmarks/` (raiz)** — caminhos obsoletos (`industrial_eval.py` aponta
  para um `build/` que não existe mais). **`archive`**.
- **`tests/` (raiz)** — **dividir**. A metade Pantheon vai para `legacy/`. A
  metade que testa o produto (`test_cpu_gpu_parity.py`,
  `test_determinism.py`, `test_save_load_equivalence.py`, `integration/`,
  `fuzzing/`) deve ser **salva**: são testes do produto suportado, mortos
  apenas por uma linha de `sys.path` obsoleta. Corrigir os caminhos e mover
  para `OXN/nsos/tests/`, onde o CTest alcança. **É a limpeza de maior valor
  deste grupo.**

---

### `legacy/` — **sem mudança; a política está funcionando**

4.044 LOC. É a única árvore cujo status declarado, regras documentadas e
estado real concordam integralmente. Ausente de todo CMakeLists, de toda CI e
de ambos os Dockerfiles. É o destino de `bindings/`, `pantheon/`, `tools/`,
`benchmarks/` e da metade Pantheon de `tests/`.

---

## Tabela de decisões

| Árvore | LOC | Código real? | Alegações sustentadas? | **Decisão** |
|---|---:|---|---|---|
| `KernelOpen/` | 5.619 | Escalonador sim; módulos exóticos não | **Não** — telemetria fabricada (constante + RNG) | **extract** + remover alegações |
| `CHRASS/` | 7.754 | **Sim** — AVX2 + radix heap sérios | Não — 57 relaxações em 1M nós; RSA = grafo caminho | **keep incubating** (3 condições) |
| `CART/` | 3.148 | **Sim** — 14 algoritmos PEFT | Não — 113× não reproduzido, admitido pela própria árvore | **keep incubating** (1 sprint) |
| `OXB/` | 3.968 | **Sim** — portável e defensivo | Reauditadas honestamente para baixo | **keep incubating** → promote |
| `research/` | 5.657 | **Sim** — melhor metodologia do repo | Sim, com artefatos | **keep incubating** permanente |
| `experimental/` | 4.505 | **Sim** — harness do produto | Nenhuma feita | **versionar imediatamente** |
| `hardware/` | 216 | Parcial — mem ctrl é mock | Não | **extract** ou delete |
| `bindings/` | 51 | Binding correto sobre stubs | N/A | **archive** |
| `pantheon` | 1.444 | **Não** — 6 de 7 `.cpp` vazios | N/A | **archive** |
| `tools/` | 276 | Utilitários finos | N/A | **archive** |
| `scripts/` raiz | 1.315 | Misto | N/A | **triagem** (manter boundary check) |
| `benchmarks/` raiz | 1.413 | Sim, caminhos obsoletos | N/A | **archive** |
| `tests/` raiz | 3.278 | Metade testa o produto | N/A | **dividir** — salvar metade NSOS |
| `legacy/` | 4.044 | Congelado por design | N/A | **sem mudança** |

---

## Se apenas três coisas forem feitas

1. **Versionar `experimental/`** — 4.505 linhas de avaliação do produto,
   incluindo um harness de 1.743 linhas mais completo que o gate oficial,
   existem em um único disco e em lugar nenhum além dele.
2. **Remover a telemetria fabricada do `KernelOpen`** e o enquadramento RSA do
   `CHRASS` antes de qualquer publicação. Um leitor que tome
   `CHRASS/README.md` ao pé da letra conclui que existe capacidade
   criptográfica que não existe.
3. **Ligar a bateria de 16 testes do `OXB` ao CTest e à CI** — é a única
   promoção genuinamente próxima.

---

## Regras transversais

- **Sem back-reference do produto** — nada sob `OXN/nsos/` ou
  `modules/oxtamem/` pode importar destas árvores. Hoje a regra é respeitada
  no código, mas **não é verificada**; `scripts/check_project_boundary.py`
  valida apenas estrutura de JSON e existência de caminhos. Implementar a
  checagem de imports é trabalho pendente.
- **Sem promoção silenciosa** — promoção é PR único atualizando
  `PROJECT_BOUNDARY.json`, `PRODUCT.md` e este arquivo juntos.
- **Arquivamento é reversível** — mover para `legacy/` não apaga histórico.
