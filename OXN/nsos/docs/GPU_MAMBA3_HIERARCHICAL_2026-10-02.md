# Mamba-3: prefixo e sufixo hierárquicos entre chunks

`NSOS_MAMBA3_GPU_PROVIDER=flash_fp32_hierarchical_v1` seleciona o novo scan
hierárquico. A seleção é explícita; o padrão continua `dense_reference`.
Projeções usam uma seleção independente: `exact_fp32`, `rdna_bf16_v1` ou
`rdna_fp16_v1` em `NSOS_MAMBA3_PROJECTION_PROVIDER`.

| Provider | Transporte de estado entre chunks | Backward |
|---|---|---|
| `parallel_fp32_v1` | Carry sequencial | Histórico completo |
| `flash_fp32_v1` | Carry sequencial | Replay por fronteiras |
| `flash_fp32_replay_lds_v2` | Carry sequencial | Replay LDS com halos escalares explícitos |
| `flash_fp32_hierarchical_v1` | Árvore de pares afins de aridade 32 | Sufixo hierárquico e o mesmo replay LDS com halos |

Cada chunk de 32 tokens produz um resumo afim `(a,b)`, que representa
`h -> a*h+b`. Um scan Hillis–Steele calcula os prefixos inclusivos dentro de
grupos de 32 resumos. Grupos maiores produzem resumos para o próximo nível;
uma passagem descendente aplica o prefixo exclusivo do grupo anterior com
`fmaf`. O backward usa a ordem reversa de chunks e publica o sufixo aplicado
ao adjoint final fornecido pelo chamador.

A árvore termina quando seu nível superior contém no máximo 32 elementos.
Para `Q=ceil(S/32)`, cada célula PN reserva
`Q + ceil(Q/32) + ...` elementos por buffer, incluindo somente os níveis
necessários até essa condição. Dois buffers FP32 pertencem à tape e são
reutilizados no backward depois do forward. Isso não altera o histórico primal.

A nova associação FP32 difere da associação do carry sequencial. Comparações
com a referência usam os limites numéricos existentes; não se exige igualdade
bit a bit entre esses dois algoritmos. A fase continua na ordem serial original
`serial_token_fp32_wrap_v1`. Os halos do replay continuam sendo os resultados
escalares de seus chunks, e não aproximações obtidas das fronteiras da árvore.

Identidades de execução registram prefixo, sufixo, aridade, propriedade do
scratch, aritmética das fronteiras e novo layout. A política de checkpoint é
`retain_mamba3_hier32_tile32_boundary_replay_lds_v1`. Sidecars dos providers
anteriores são rejeitados pelo novo provider. Todos os consumidores e o módulo
Python precisam de rebuild porque `Trace` ganhou campos.

## Validação e medição

`test_gpu_mamba3_parallel` cobre forward, estados finais, VJPs de todos os
parâmetros e dos estados iniciais, prefixos vazios/parciais, padding NaN,
fronteiras de níveis e rejeições de propriedade/publicação. Inclui sequências
1024, 1025 e 32769; mantém os checks bit a bit entre Flash v1/v2 existentes.
`test_gpu_mamba3_scan_identity` verifica a seleção, identidade e rejeição de
sidecars incompatíveis. Também testa retomada dentro da mesma identidade em
SISO S33 e MIMO4 S1025: após três commits, exige pesos, loss e momentos de Adam
bit a bit iguais à execução contínua. A comparação usa o estado autoritativo
exportado pela API de checkpoint na lane do proprietário; os mapas host sem
sincronização podem conter momentos antigos. A rejeição de seleção otimizada
em CPU permanece.

```powershell
cmake --build OXN/nsos/build-gm-cpu --config Release
cmake --build OXN/nsos/build-gm-hip-gpuopt --config Release
ctest --test-dir OXN/nsos/build-gm-cpu --output-on-failure
ctest --test-dir OXN/nsos/build-gm-hip-gpuopt --output-on-failure -R 'test_gpu_mamba3_parallel|test_gpu_mamba3_scan_identity'
```

Essa implementação elimina o carry sequencial entre chunks do estado SSM e de
seu adjoint. A fase e o replay local continuam tendo suas dependências originais.
Lançamentos extras e scratch podem prejudicar sequências curtas. Desempenho,
VRAM e qualidade precisam de medições novas com a identidade desse provider;
recibos Flash v1/v2 não o certificam. A comparação deve reportar configuração,
precisão, sementes, warmup, contadores nativos e custo de checkpoint.

Na RX 7600, a comparação de camada com projeções FP32 e alocação explícita em
device passou os gates numéricos em 56 processos. Três pares de processos por
geometria mostraram tempo de forward/backward aproximadamente 11% maior em
D768/S512 e 21% maior em D128/S1057 que Flash LDS v2. Esses resultados não medem
o Trainer completo nem justificam promover o provider a padrão. O transporte
hierárquico está correto, mas seus lançamentos e scratch ainda precisam de
otimização orientada por perfil.
