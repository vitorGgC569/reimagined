"""
Benchmark de throughput por provedor GPU do Mamba-3.
Mede steps/s e tok/s para os 4 provedores com S=64 tokens, 20 steps de forward.
Uso: python tests/bench_mamba3_provider_throughput.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build-gm-hip'))
import nsos_ext as nsos  # noqa: E402

# ── Parâmetros ──────────────────────────────────────────────────────────────
B        = 1
S        = 64
DM       = 128   # d_model — deve ser divisível por n_heads
N_HEADS  = 4     # 32 dims por head
VS       = 512
N_WARMUP = 5
N_STEPS  = 20

PROVIDERS = [
    # (rótulo, env-key, env-val, proj-env-val)
    ('dense_reference',  'dense_reference',  'exact_fp32'),
    ('parallel_fp32_v1', 'parallel_fp32_v1', 'exact_fp32'),
    ('flash_fp32_v1',    'flash_fp32_v1',    'exact_fp32'),
    ('wmma_fp16',        'dense_reference',  'rdna_fp16_v1'),
]

# ── Helpers ──────────────────────────────────────────────────────────────────
def set_providers(scan_val, proj_val):
    os.environ['NSOS_MAMBA3_GPU_PROVIDER']           = scan_val
    os.environ['NSOS_MAMBA3_PROJECTION_PROVIDER']    = proj_val


def make_model():
    cfg = nsos.ModelConfig()
    cfg.num_layers                = 2
    cfg.d_model                   = DM
    cfg.n_heads                   = N_HEADS
    cfg.n_kv_heads                = N_HEADS
    cfg.vocab_size                = VS
    cfg.architecture_schema_version = 3   # obrigatório para Mamba-3
    cfg.mamba3_schema_version     = 1
    cfg.mamba3_enabled            = True
    cfg.attention_period          = 9999  # valor alto: nenhuma das 2 camadas será atenção
    return nsos.JambaModel(cfg, nsos.Device.GPU)


# ── Benchmark ────────────────────────────────────────────────────────────────
def main():
    ids = list(range(S)) * B

    print(f'\nGPU Throughput Benchmark  B={B}  S={S}  DM={DM}  N_STEPS={N_STEPS}')
    print(f'{"Provider":<22} {"steps/s":>10} {"tok/s":>12} {"ms/step":>10}')
    print('-' * 58)

    results = {}

    for label, scan_val, proj_val in PROVIDERS:
        try:
            set_providers(scan_val, proj_val)
            model = make_model()

            # warmup — deixa o kernel JIT compilar e os caches aquecidos
            for _ in range(N_WARMUP):
                model.forward_ids(ids)

            t0 = time.perf_counter()
            for _ in range(N_STEPS):
                model.forward_ids(ids)
            elapsed = time.perf_counter() - t0

            sps = N_STEPS / elapsed
            tps = sps * S * B
            mss = elapsed / N_STEPS * 1000
            results[label] = sps
            print(f'{label:<22} {sps:>10.2f} {tps:>12.1f} {mss:>10.2f}')

            del model

        except Exception as exc:
            print(f'{label:<22} ERRO: {exc}')

    if results:
        base = results.get('dense_reference', 1.0)
        print()
        print('Speedup vs dense_reference:')
        for k, v in results.items():
            print(f'  {k:<22}: {v / base:.2f}x')


if __name__ == '__main__':
    main()
