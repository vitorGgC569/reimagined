# Relatorio de auditoria e mapa de MVP - 2026-04-26

## Veredito executivo

O projeto tem um nucleo tecnico aproveitavel para um MVP, mas o escopo precisa ser reduzido e endurecido. O caminho mais seguro e tratar `OXN/nsos` como produto principal, com MVP de inferencia local/autenticada em CPU, empacotamento de modelo e API HTTP/CLI/Python. GPU, distribuicao, OxtaMem como servico publico, TTT, Lean/formal verification e varios scripts antigos ainda devem ficar como experimentais.

Estado observado:

- Core CPU de `OXN/nsos` esta em bom estado relativo: `build-ci-local` passou 20/20 testes.
- Build oficial documentado como `build_v1` esta inconsistente: 10/20 testes falharam ou nem tinham executavel.
- Build CUDA 12.9 passou 19/20, mas `test_gpu_parity` segfaultou.
- Rust/OxtaMem compila e testes basicos passam, mas `cargo clippy -D warnings` falha em pontos de FFI e qualidade.
- Binding Python `nsos_ext.generate()` consegue importar no build CPU, mas pode retornar bytes invalidos como UTF-8.
- Dockerfiles estao quebrados para execucao e pouco reprodutiveis.
- Existem muitos componentes simulados, mocks, stubs e documentos que prometem mais do que o codigo atual entrega.

## Escopo real do projeto

O arquivo `PROJECT_SCOPE_STATUS.md` indica que o core oficial hoje e:

- `OXN/nsos`: runtime C++20/CUDA, API HTTP, CLI, bindings Python, ferramentas de modelo e testes.
- `modules/oxtamem`: memoria semantica/geodesica em Rust/Python, integrada apenas parcialmente.

Os demais blocos (`KernelOpen`, `CHRASS`, `CART`, `OXB`, `hardware`, scripts antigos e demos) devem ser considerados pesquisa, legado ou integracao futura ate passarem pelo mesmo nivel de build, teste e documentacao.

## Evidencias de verificacao

Comandos executados:

- `ctest --test-dir OXN\nsos\build-ci-local -C Release --output-on-failure`
  - Resultado: 20/20 testes passaram.
  - Observacao: build CPU, `NSOS_ENABLE_CUDA=OFF`, `NSOS_BUILD_OXTAMEM=OFF`.
- `ctest --test-dir OXN\nsos\build_v1 -C Release --output-on-failure`
  - Resultado: 10/20 falharam.
  - Falhas por executaveis ausentes: `test_sanity`, `test_bitnet_integrated`, `test_mamba2`, `test_kan`, `test_ttt_layer_kernel`, `test_jamba`, `test_holographic_full`, `test_ultra_integration`, `test_swarm_orchestration`.
  - Falha real: `test_gpu_parity` com erro de cuBLAS.
- `ctest --test-dir OXN\nsos\build_cuda129 -C Release --output-on-failure`
  - Resultado: 19/20 passaram.
  - Falha: `test_gpu_parity` com segmentation fault.
- `cargo test --manifest-path modules\oxtamem\oxta_engine\Cargo.toml --all-targets`
  - Resultado: passou, mas a cobertura e minima.
- `cargo clippy --manifest-path modules\oxtamem\oxta_engine\Cargo.toml --all-targets -- -D warnings`
  - Resultado: falhou por problemas de FFI unsafe, `std::io::Error` e complexidade de tipo.
- `python -m compileall -q ...`
  - Resultado: scripts Python principais compilaram.
- `python -m pytest OXN\tests -q` e `python -m pytest tests -q`
  - Resultado: bloqueado, `pytest` nao instalado.
- Import de `nsos_ext` pelo build CPU:
  - Importou e `load_model('', cfg)` retornou true.
  - `generate('hi', 1, 0.0)` falhou com `UnicodeDecodeError`.

## Achados criticos

### 1. API HTTP pode rodar sem autenticacao efetiva

Arquivos:

- `OXN/nsos/include/http_api_server.h`
- `OXN/nsos/src/api_server.cpp`
- `OXN/nsos/src/http_api_server.cpp`

O token de autenticacao e opcional. Se `--auth-token` ou `NSOS_API_TOKEN` nao forem definidos, `request_requires_auth` libera os endpoints. O default de host e `127.0.0.1`, mas a CLI aceita `--host 0.0.0.0`. Isso permite expor geracao, treino online, batch e escrita de pack sem autenticacao caso alguem publique a porta.

Impacto:

- Inferencia e treino remotos sem autorizacao.
- DoS por computacao prolongada.
- Alteracao/corrupcao do estado do modelo por endpoints de treino.
- Escrita de arquivos por `/pack`.

Correcao para MVP:

- Falhar o startup se `host` nao for loopback e nao houver token.
- Exigir token para todos os endpoints exceto `/health`, se explicitamente configurado.
- Desabilitar `/train-*` e `/pack` por default; liberar apenas com flag admin.

### 2. `/pack` aceita diretorio arbitrario

Arquivos:

- `OXN/nsos/src/http_api_server.cpp`
- `OXN/nsos/src/nsos_sdk.cpp`

O endpoint `/pack` recebe `directory` e chama `engine.save_model_pack(directory)`. Isso grava arquivos como `model.nsos.bin`, `tokenizer.nsos`, `config.nsos`, `edge_linear.nsos` e `manifest.nsos` no caminho informado.

Impacto:

- Escrita arbitraria dentro das permissoes do processo.
- Com API sem token, vira falha critica.

Correcao para MVP:

- Remover o endpoint do build publico ou exigir modo admin.
- Restringir a escrita a um diretorio configurado de artefatos.
- Canonicalizar caminho e rejeitar path traversal.

### 3. Servidor OxtaMem abre rede sem autenticacao e sem limites

Arquivos:

- `modules/oxtamem/oxta_engine/src/server.rs`
- `modules/oxtamem/oxta_engine/src/engine.rs`

O servidor RESP usa bind em `0.0.0.0:{port}` por default, sem autenticacao, TLS, limite de conexoes, limite de frame, timeout ou controle de memoria. O parser usa `Vec::with_capacity(count)` com valor vindo do cliente.

Impacto:

- Exposicao de armazenamento/memoria.
- DoS por conexoes lentas, frames grandes ou contagens artificiais.
- Crescimento de arquivo controlado por configuracao de path/size.

Correcao para MVP:

- OxtaMem nao deve ser servico publico no MVP.
- Se incluido, bind default deve ser `127.0.0.1`, com token, limite de frame, timeout e limite de conexoes.

### 4. GPU parity esta instavel

Arquivos:

- `OXN/nsos/tests/test_gpu_parity.cpp`
- componentes CUDA chamados pelo teste

O build CUDA 12.9 tem `test_gpu_parity` com segmentation fault. O build `build_v1` tambem falhou nesse teste com erro de cuBLAS.

Impacto:

- GPU nao pode ser prometida como caminho de producao.
- Qualquer benchmark ou demo GPU pode quebrar de forma nao deterministica.

Correcao para MVP:

- MVP deve declarar GPU como experimental.
- Isolar o trecho do teste que falha: add/matmul/rmsnorm, BitLinear, Mamba streaming e Jamba batch.
- Adicionar sanitizacao CUDA, logs de device/cublas e teste menor por kernel.

### 5. Saida Python/HTTP pode conter UTF-8 invalido

Arquivos:

- `OXN/nsos/src/bindings.cpp`
- `OXN/nsos/src/tokenizer.cpp`
- `OXN/nsos/src/nsos_sdk.cpp`

O tokenizer pode decodificar bytes arbitrarios. O binding Python expoe `InferenceEngine.generate` como `std::string`. No teste manual, `engine.generate('hi', 1, 0.0)` gerou `UnicodeDecodeError`.

Impacto:

- Binding Python quebra em uso basico.
- JSON HTTP pode ser comprometido se strings invalidas escaparem.

Correcao para MVP:

- Garantir UTF-8 valido na fronteira HTTP/Python.
- Usar replacement character, escaping, bytes explicitos ou token IDs.
- Adicionar teste Python para `generate` com saida serializavel.

## Achados altos

### 6. API sem timeout de socket e vulneravel a conexoes lentas

`http_api_server.cpp` usa `recv` bloqueante para ler request/header/body e so aplica rate limit depois da leitura. Nao foi encontrado uso de `SO_RCVTIMEO`, `SO_SNDTIMEO`, `select`, `poll` ou limite de tempo por request.

Correcao:

- Timeouts de leitura/escrita.
- Limite de tempo por request.
- Rate limit e verificacoes basicas antes de aceitar body grande.

### 7. Parametros de computacao nao tem tetos seguros

Endpoints de geracao e treino aceitam valores como `max_tokens`, numero de prompts, `steps`, `epochs`, `batch_size`, `seq_len` e `max_steps` sem limites superiores consistentes.

Correcao:

- Definir limites no `ApiServerConfig`.
- Rejeitar requests fora do envelope.
- Separar endpoints mutaveis de endpoints de inferencia.

### 8. Desserializacao insegura no SDK Python OxtaMem

Arquivo:

- `modules/oxtamem/python/oxta_mem/sdk.py`

O fallback usa `pickle.loads`. Tambem ha usos de `torch.load` e `np.load` sobre bytes recuperados. Isso e seguro apenas quando a origem e totalmente confiavel.

Correcao:

- Remover pickle do caminho default.
- Aceitar apenas formatos tipados e seguros para MVP: texto, JSON, arrays com formato controlado.
- Documentar explicitamente quando um backend e trusted-only.

### 9. Packs de modelo/tokenizer/edge nao sao suficientemente endurecidos

Arquivos:

- `OXN/nsos/src/nsos_sdk.cpp`
- `OXN/nsos/src/jamba.cpp`
- `OXN/nsos/src/tokenizer.cpp`

Riscos:

- Manifest pode referenciar caminhos relativos fora do pack.
- Checksums sao integridade, nao autenticidade.
- Config maliciosa pode tentar alocar estruturas grandes.
- Edge pack le contagens e aloca vetores antes de validar limites globais.
- Tokenizer pack nao tem limite forte de linhas/tokens/ids.
- Save de pack nao e atomico.

Correcao:

- Canonicalizar todos os caminhos dentro do pack root.
- Limites por arquivo, tensor, vocab, camada e dimensao.
- Escrita por arquivo temporario + rename atomico.
- Manifest versionado e, se houver distribuicao externa, assinatura.

### 10. Dockerfiles quebrados e pouco reprodutiveis

Arquivos:

- `Dockerfile`
- `Dockerfile.industrial`
- `.dockerignore`

Problemas:

- `Dockerfile` usa `CMD ["python3", "OXN/scripts/train_industrial.py"]`, mas a stage runtime nao copia `OXN/scripts`.
- `Dockerfile.industrial` copia `/src/OXN/scripts` para `/app/scripts`, mas usa `CMD ["python3", "scripts/serve_oxn.py"]`; esse arquivo nao existe nesse caminho esperado.
- Dependencias Python nao estao pinadas.
- Imagens rodam como root.
- `.dockerignore` nao exclui bem builds e artefatos aninhados.

Correcao:

- Ter um unico Dockerfile de MVP.
- Copiar apenas runtime necessario.
- Usar usuario nao-root.
- Pin de dependencias.
- Smoke test da imagem no CI.

### 11. Gate de testes incompleto

Problemas:

- Muitos testes C++ em `OXN/nsos/tests` nao estao ligados ao CMake.
- `pytest` nao esta instalado, entao testes Python nao rodam.
- CI CPU desliga CUDA e OxtaMem.
- CI nao roda `cargo test`, `cargo clippy`, Docker build, ou auditoria de dependencias.

Correcao:

- Definir matriz minima de CI do MVP.
- Incluir Rust e Python no gate.
- Arquivar testes antigos ou conecta-los ao CMake.
- Garantir que a build documentada seja a mesma validada.

## Achados medios

### 12. Documentacao superestima maturidade

Arquivos e areas:

- `OXN/README.md`
- `OXN/nsos/docs/*`
- root `README.md`

O root README so contem o nome. O README de `OXN` promete recursos como MPI real, Lean/self-healing e maturidade industrial que nao batem com o estado validado. A documentacao mais honesta esta em `OXN/nsos/docs/NSOS_VALIDATION_STATUS.md`.

Correcao:

- Trocar o root README por um guia oficial.
- Criar matriz de suporte: estavel, beta, experimental, legado.
- Remover ou arquivar claims que ainda nao tem teste.

### 13. Scripts com caminhos hardcoded

Exemplos:

- `start_model_chat.ps1`
- `start_small_overnight.ps1`

Ha defaults como `C:\Users\Oxta\Desktop\reimagined-main` e CUDA `v12.9`.

Correcao:

- Resolver caminhos relativos ao script.
- Permitir override por env vars.
- Validar prerequisitos com erro claro.

### 14. Repositorio sem metadados Git no workspace

`git status` falhou porque o diretorio atual nao e um repositorio Git. Isso impede diff confiavel, blame, versionamento de release e rastreabilidade.

Correcao:

- Trabalhar em checkout Git real.
- Criar tag/release do MVP.

## Codigo incompleto, simulado ou legado

Itens que nao devem entrar como promessa de MVP sem revalidacao:

- `OXN/nsos/inference.py`: blueprint experimental; classes levantam `RuntimeError`.
- `OXN/nsos/data_pipeline.py`: blueprint experimental; metodos nao implementados.
- `OXN/nsos/evaluation.py`: blueprint experimental; metodos nao implementados.
- `OXN/nsos/python/nsos/shield.py`: `contiguous_check` vazio, `state_dict` TODO e fallback nao implementado.
- `OXN/nsos/src/chat.cpp`: embeddings e tokens simulados; contexto de demo.
- `OXN/nsos/src/fabric.cpp`: MPI send/recv/bcast placeholder; CMake usa `fabric_v2.cpp`, mas o arquivo antigo ainda confunde.
- `OXN/nsos/src/server.cpp`: servidor demo antigo com parsing/embedding/decode dummy.
- `OXN/nsos/src/cuda/persistent_kernel.cu`: kernel persistente stub/minimo.
- `OXN/nsos/include/mpi_mock.h`: mock explicito.
- `OXN/nsos/include/lean_integration.h`: verificacao formal mock/stub.
- `serve_oxn.py`: servidor antigo sem auth, bind em todas as interfaces, decode/token dummy.
- `modules/oxtamem/python/oxta_mem/sdk.py`: `recall_history` Redis marcado como nao implementado, apesar de o servidor Rust ter comando `RECALL`.
- `modules/oxtamem/python/oxta_mem/langchain.py`: mocks quando LangChain nao esta instalado.
- `modules/oxtamem/oxta_engine/python/langchain_retriever.py`: mocks quando LangChain nao esta instalado.
- `modules/oxtamem/nn/geodesic_transformer.py`: `MockClient` e loop simulado.

## MVP recomendado

Nome sugerido: **NSOS Local/Authenticated CPU Inference MVP**.

Objetivo:

Entregar uma versao executavel e demonstravel que carrega um model pack, faz inferencia por CLI/API/Python, salva pack de forma controlada, tem testes reproduziveis e documentacao honesta.

Dentro do MVP:

- `OXN/nsos` como produto principal.
- CPU float e CPU packed.
- API HTTP local ou autenticada.
- CLI.
- Binding Python com saida UTF-8 segura.
- Model pack versionado com checksum e limites.
- Test suite CPU passando.
- Docker runtime funcional.
- Benchmark/scorecard minimo.

Fora do MVP:

- GPU como caminho obrigatorio.
- Servidor OxtaMem exposto em rede.
- Distribuicao/MPI real.
- TTT como recurso de produto.
- Lean/formal verification como garantia.
- Scripts e servidores legados.
- Claims de "industrial ready" sem gate automatizado.

## Checklist obrigatorio para MVP

1. Canonicalizar build oficial
   - Criar/validar um build unico, por exemplo `build-mvp`.
   - Atualizar README para apontar so para esse build.
   - Remover dependencia de `build_v1` se ele nao for regenerado.

2. Endurecer API HTTP
   - Token obrigatorio fora de loopback.
   - `/health` como unica rota anonima por default.
   - `/train-*` e `/pack` atras de flag admin.
   - Tetos para `max_tokens`, batch size, prompts, steps, epochs, body e tempo.
   - Timeouts de socket.
   - Rate limit antes do processamento pesado.

3. Corrigir fronteira UTF-8
   - Garantir que HTTP JSON e Python retornem strings validas.
   - Adicionar teste de import/load/generate via Python.

4. Endurecer packs
   - Canonicalizar paths dentro do pack root.
   - Limites de tamanho, vocab, tensores e dimensoes.
   - Escrita atomica.
   - Manifest com versao explicita.

5. Fechar gate de teste
   - `ctest` CPU 20/20.
   - Teste Python com `pytest` instalado.
   - `cargo test` e `cargo clippy -D warnings`.
   - Docker build + smoke test.
   - Lista clara de testes antigos arquivados ou religados.

6. Corrigir Docker
   - Um Dockerfile oficial para MVP.
   - CMD apontando para binario/script existente.
   - Dependencias pinadas.
   - Usuario nao-root.
   - `.dockerignore` cobrindo artefatos aninhados.

7. Limpar documentacao
   - Root README operacional.
   - Matriz de maturidade por componente.
   - Guia de execucao local.
   - Guia de seguranca minima.
   - Remover claims nao validados.

8. Preparar demo/release
   - Artefato de model pack pequeno.
   - Script de smoke test.
   - Benchmark JSON.
   - Exemplos `curl`, CLI e Python.

## Fases sugeridas

### Fase 0 - Congelamento de escopo

- Declarar `OXN/nsos` como unica superficie de MVP.
- Marcar legado/experimental no README.
- Remover do caminho feliz os scripts antigos.

### Fase 1 - Hardening minimo

- API auth/limites/timeouts.
- UTF-8 seguro.
- Pack path hardening.
- Docker funcional.

### Fase 2 - Validacao

- CI local reproduzivel.
- Testes Python e Rust no gate.
- GPU documentada como experimental ate `test_gpu_parity` estabilizar.

### Fase 3 - Empacotamento

- Model pack pequeno.
- CLI/API/Python smoke tests.
- Benchmark basico e relatorio de limites conhecidos.

### Fase 4 - Documentacao e release

- README de produto.
- Guia de instalacao.
- Guia de seguranca.
- Tag/release em Git real.

## Riscos residuais aceitos para o MVP

Se o MVP for limitado a loopback/autenticado e CPU:

- GPU instavel pode ser aceito como nao suportado.
- Distribuicao pode ser aceita como fora de escopo.
- OxtaMem pode ser aceito como biblioteca experimental, sem servidor publico.
- Packs sem assinatura podem ser aceitos se a documentacao exigir origem confiavel.

Nao deve ser aceito:

- API publica sem token.
- `/pack` arbitrario em build publico.
- Python/HTTP quebrando com UTF-8 invalido.
- Docker oficial que nao inicia.
- README prometendo capacidades nao validadas.

## Proximas acoes tecnicas recomendadas

1. Implementar fail-fast da API quando `host` nao for loopback e token estiver vazio.
2. Adicionar limites de request/geracao/treino no `ApiServerConfig`.
3. Tornar `/pack` e `/train-*` opt-in por flag admin.
4. Corrigir retorno UTF-8 de `generate` nas fronteiras Python e HTTP.
5. Corrigir `cargo clippy -D warnings` em OxtaMem.
6. Criar Dockerfile oficial de MVP e smoke test.
7. Atualizar root README com matriz de maturidade.
8. Investigar `test_gpu_parity` separando os subtestes por kernel.

