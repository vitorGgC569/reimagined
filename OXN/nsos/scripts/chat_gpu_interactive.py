import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

root = Path(r'C:\Users\vitor\OneDrive\Desktop\OGrandeOxta\reimagined')
sys.path.insert(0, str(root / 'OXN' / 'nsos' / 'scripts'))
import train_curriculum as tc
import train_ptbr_conversational as pt

checkpoint = root / 'OXN' / 'nsos' / 'artifacts' / 'ptbr_conversational' / 'pilot' / 'runs' / 'main' / 'checkpoints' / 'generations' / 'step-000000229537-complete'
build_dir = root / 'OXN' / 'nsos' / 'build-validation-hip'
pack_path = root / 'OXN' / 'nsos' / 'artifacts' / 'ptbr_conversational' / 'pilot' / 'packs' / 'pack_manifest.json'

print("======================================================================")
print(" 🚀 INICIANDO CHAT COM MODELO IA PORTUGUÊS-BR (GPU FORÇA TOTAL) ")
print(" Checkpoint Definitivo: step-000000229537-complete")
print("======================================================================")

nsos = tc.load_nsos(build_dir)
tokenizer = nsos.Tokenizer()
tokenizer.load(str(checkpoint / 'tokenizer.nsos'))
config = pt.build_model_config(nsos, dict(pt.PRESETS['pilot']), int(tokenizer.vocab_size), nsos.Device.GPU)

print("\n[GPU] Alocando modelo e pesos na GPU...")
t0 = time.perf_counter()
model = nsos.JambaModel(config, nsos.Device.GPU)
model.to(nsos.Device.GPU)
model.load(str(checkpoint / 'model.bin'), True)
dt_load = time.perf_counter() - t0
print(f"[GPU] Modelo pronto na GPU em {dt_load:.2f}s!\n")

eos = int(json.loads(pack_path.read_text(encoding='utf-8'))['tokenizer']['eos_token_id'])
system_prefix = '<|bos|><|system|>\nVocê é um assistente conversacional brasileiro. Responda somente em português do Brasil, com clareza, honestidade e naturalidade. Quando não souber algo, diga que não sabe em vez de inventar.\n<|user|>\n'
system_suffix = '\n<|assistant|>\n'

print("Digite sua mensagem para a IA e pressione Enter.")
print("Para sair do chat, digite /sair ou /exit.\n")

while True:
    try:
        user_input = input("\nVocê > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\nSaindo do chat...")
        break

    if not user_input:
        continue
    if user_input.lower() in {'/sair', '/exit', 'exit', 'quit'}:
        print("Chat encerrado.")
        break

    prompt = system_prefix + user_input + system_suffix

    t_start = time.perf_counter()
    generated = tc.greedy_generate(nsos, model, tokenizer, prompt, 128, eos)
    t_end = time.perf_counter() - t_start
    n_tokens = len(tokenizer.encode(generated))
    tok_per_sec = n_tokens / t_end if t_end > 0 else 0

    print(f"\nIA (GPU) > {generated}")
    print(f"\n[Métricas GPU] {n_tokens} tokens em {t_end:.3f}s ({tok_per_sec:.1f} tok/s)")
