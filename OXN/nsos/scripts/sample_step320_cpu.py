import json, sys, time
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

root = Path(r'C:\Users\vitor\OneDrive\Desktop\OGrandeOxta\reimagined')
sys.path.insert(0, str(root / 'OXN' / 'nsos' / 'scripts'))
import train_curriculum as tc
import train_ptbr_conversational as pt

checkpoint = root / 'OXN' / 'nsos' / 'artifacts' / 'ptbr_conversational' / 'pilot' / 'runs' / 'main' / 'checkpoints' / 'generations' / 'step-000000013740-exception-safe-point'
build_dir = root / 'OXN' / 'nsos' / 'build-codex-hip'
pack_path = root / 'OXN' / 'nsos' / 'artifacts' / 'ptbr_conversational' / 'pilot' / 'packs' / 'pack_manifest.json'

nsos = tc.load_nsos(build_dir)
tokenizer = nsos.Tokenizer()
tok_file = checkpoint / 'tokenizer.nsos'
if not tok_file.is_file():
    tok_file = root / 'OXN' / 'nsos' / 'artifacts' / 'ptbr_conversational' / 'pilot' / 'tokenizer.nsos'
tokenizer.load(str(tok_file))
config = pt.build_model_config(nsos, dict(pt.PRESETS['pilot']), int(tokenizer.vocab_size), nsos.Device.CPU)

model = nsos.JambaModel(config, nsos.Device.CPU)
model.to(nsos.Device.CPU)
model.load(str(checkpoint / 'model.bin'), True)

eos = int(json.loads(pack_path.read_text(encoding='utf-8'))['tokenizer']['eos_token_id'])
system_prefix = '<|bos|><|system|>\nVocê é um assistente conversacional brasileiro. Responda somente em português do Brasil, com clareza, honestidade e naturalidade. Quando não souber algo, diga que não sabe em vez de inventar.\n<|user|>\n'
system_suffix = '\n<|assistant|>\n'

prompts = [
    "Hello!",
    "Can you help me with a math problem?",
    "What is the capital of France?",
    "If John has 5 apples and buys 3 more, how many apples does he have?",
    "Explain what technology is.",
    "Who discovered gravity?",
    "Write a short paragraph about science."
]

print("======================================================================")
print(" 🧪 AMOSTRAGEM CPU - REAL CHECKPOINT ATUAL (STEP 13.740)")
print("======================================================================\n")
for p in prompts:
    full_p = system_prefix + p + system_suffix
    gen = tc.greedy_generate(nsos, model, tokenizer, full_p, 64, eos)
    print(f"Prompt  : {p}")
    print(f"Resposta: {gen}")
    print("-" * 50)
