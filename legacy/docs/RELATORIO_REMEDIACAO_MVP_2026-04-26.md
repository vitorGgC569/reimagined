# Relatorio de remediacao MVP - 2026-04-26

## Implementado

- API HTTP endurecida:
  - token obrigatorio por padrao;
  - bind fora de loopback sem token recusado;
  - modo local sem auth somente com `--allow-unauthenticated-local`;
  - `/train-*` e `/pack` atras de `--enable-admin-endpoints`;
  - limites para geracao, batch, treino, JSON depth, body/header e timeout de socket;
  - `/pack` restrito a `--pack-root` por padrao.
- Fronteira UTF-8 corrigida:
  - tokenizer sanitiza decode invalido com replacement character;
  - JSON escape nao emite bytes UTF-8 invalidos;
  - smoke Python cobre `InferenceEngine.generate()`.
- Packs endurecidos:
  - manifest precisa de `format=nsos-pack-v2` e `version=2`;
  - paths do manifest rejeitam absolutos e `..`;
  - limites de tamanho por config/tokenizer/weights/edge pack;
  - config de pack validada antes de construir modelo;
  - escrita de arquivos de pack passa por temporario + rename.
- Edge pack endurecido:
  - vetores lidos com limites derivados da camada;
  - dimensoes invalidas rejeitadas antes de alocar.
- OxtaMem corrigido:
  - servidor RESP usa loopback por padrao;
  - auth via `AUTH`/`OXTAMEM_AUTH_TOKEN`;
  - limite de conexoes, frame, bulk string, args e timeout;
  - bind fora de loopback sem token recusado;
  - FFI marcada `unsafe` com contrato documentado;
  - metadata escrita de forma atomica;
  - arena e payloads com limites;
  - `cargo clippy -D warnings` limpo.
- Vulnerabilidade Rust corrigida:
  - `bytes` atualizado de `1.11.0` para `1.11.1`;
  - `cargo audit` limpo.
- SDK Python OxtaMem:
  - removido pickle;
  - serializer seguro para bytes, str, JSON, NumPy e Torch tensor;
  - `recall_history` via RESP implementado.
- Docker:
  - Dockerfile oficial de MVP CPU;
  - Dockerfile industrial convertido para compatibilidade segura;
  - runtime exige `NSOS_API_TOKEN`;
  - usuario nao-root;
  - dependencias Python pinadas;
  - `.dockerignore` cobre artefatos aninhados.
- CI:
  - dependencias Python pinadas em `requirements-dev.txt`;
  - compileall Python;
  - Rust `cargo test`, `cargo clippy` e `cargo audit`;
  - build Docker como gate.
- Documentacao:
  - root README substituido por guia de MVP;
  - `OXN/README.md` substituido por matriz de maturidade;
  - `OXN/nsos/README.md` atualizado para `build-mvp` e contrato seguro;
  - OxtaMem README atualizado sem claim de servico publico.
- Legado/mocks:
  - `serve_oxn.py` desabilitado para nao expor servidor mock;
  - `shield.py` implementa checks basicos e `state_dict`;
  - GPU parity test agora tem casos isolados por `NSOS_GPU_PARITY_CASE`.

## Validacoes executadas

- `cmake -S .\OXN\nsos -B .\OXN\nsos\build-mvp ...`
- `cmake --build .\OXN\nsos\build-mvp --config Release`
- `ctest --test-dir .\OXN\nsos\build-mvp -C Release --output-on-failure`
  - Resultado: 21/21 passaram.
- `python -m compileall -q ...`
  - Resultado: passou.
- `python .\OXN\nsos\tests\test_python_binding_smoke.py`
  - Resultado: passou.
- `cargo test --manifest-path modules\oxtamem\oxta_engine\Cargo.toml --all-targets`
  - Resultado: passou.
- `cargo clippy --manifest-path modules\oxtamem\oxta_engine\Cargo.toml --all-targets -- -D warnings`
  - Resultado: passou.
- `cargo audit`
  - Resultado: passou apos atualizar `bytes`.
- `.\OXN\nsos\build-mvp\Release\nsos_api_server.exe --host 127.0.0.1 --port 0`
  - Resultado esperado: recusou startup sem token.

## Bloqueado localmente

- `docker build -t nsos-mvp-local .`
  - Bloqueado porque o Docker Desktop daemon nao esta rodando:
    `failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine`.
  - O Docker build esta configurado como gate no CI.

## Risco residual

- CUDA/GPU continua experimental ate o segfault do build CUDA ser depurado no
  ambiente com driver/GPU. O teste agora isola os casos, mas o MVP documentado e
  CPU-first.
