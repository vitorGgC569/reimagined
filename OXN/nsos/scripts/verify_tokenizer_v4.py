"""Quick sanity check: verify v4 tokenizer does NOT contain C++ identifier tokens."""
from __future__ import annotations
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
BUILD   = Path("C:/Users/Oxta/Desktop/reimagined-main/OXN/nsos/build-mvp/Release")

if str(BUILD) not in sys.path:
    sys.path.insert(0, str(BUILD))

import nsos_ext as nsos  # type: ignore

# Load OLD v3 tokenizer
print("=== V3 Tokenizer (C++ contaminated) ===")
tok3 = nsos.Tokenizer()
tok3.load(str(SCRIPTS / "distillation_bundle_v3" / "tokenizer_8192.ox3"))
print(f"Vocab size: {tok3.vocab_size}")
# Check if ensure_kv_cache_capac is a single token
ids_v3 = tok3.encode("ensure_kv_cache_capac")
print(f'encode("ensure_kv_cache_capac"): {len(ids_v3)} tokens → {ids_v3[:5]}')
ids_v3_hello = tok3.encode("Hello, how are you?")
print(f'encode("Hello, how are you?"): {len(ids_v3_hello)} tokens → {ids_v3_hello}')
print()

# Load NEW v4 tokenizer
print("=== V4 Tokenizer (Wikipedia-based) ===")
tok4 = nsos.Tokenizer()
tok4.load(str(SCRIPTS / "distillation_bundle_v4" / "tokenizer_8192.ox3"))
print(f"Vocab size: {tok4.vocab_size}")
ids_v4 = tok4.encode("ensure_kv_cache_capac")
print(f'encode("ensure_kv_cache_capac"): {len(ids_v4)} tokens → {ids_v4[:10]}')
ids_v4_hello = tok4.encode("Hello, how are you?")
print(f'encode("Hello, how are you?"): {len(ids_v4_hello)} tokens → {ids_v4_hello}')
ids_v4_paris = tok4.encode("Paris is the capital of France.")
print(f'encode("Paris is the capital of France."): {len(ids_v4_paris)} tokens → {ids_v4_paris}')

# Decode a few token IDs from v4 to see what they represent
print("\nFirst 20 learned tokens (v4):")
for i in range(256, min(276, int(tok4.vocab_size))):
    piece = tok4.decode([i])
    print(f"  [{i}] = {piece!r}")
