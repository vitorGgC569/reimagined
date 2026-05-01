import os
import json
import random
from benchmarks.common import NUM_SAMPLES, JSONL_FILE, TOON_FILE, DATA_DIR

def generate():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print(f"Generating {NUM_SAMPLES} samples...")

    # Sample Texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "To be or not to be, that is the question.",
        "Machine learning is fascinating and powerful.",
        "OxtaCore optimizes data ingestion for LLMs.",
        "Python is a great language for data science."
    ]

    with open(JSONL_FILE, 'w', encoding='utf-8') as fj, open(TOON_FILE, 'w', encoding='utf-8') as ft:
        # TOON Header (Simulated)
        ft.write(f"items[{NUM_SAMPLES}]" + "{text,score}:\n")

        for i in range(NUM_SAMPLES):
            text = random.choice(texts) * 5 # Make it longer
            score = round(random.random(), 4)

            # JSONL
            obj = {"text": text, "score": score}
            fj.write(json.dumps(obj) + "\n")

            # TOON (CSV-like: text,score)
            # Escape commas in text just in case (though our sample has none)
            safe_text = text.replace(",", ";")
            ft.write(f"{safe_text},{score}\n")

    print(f"Generated:\n - {JSONL_FILE}\n - {TOON_FILE}")

if __name__ == "__main__":
    generate()
