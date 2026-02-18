import os
import json
import random
import torch

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    print("⚠️  'datasets' library not found. Using MOCK mode.")
    HAS_DATASETS = False

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

def save_to_bin(data_list, path, tokenizer=None):
    """Encodes text to integers and saves as torch tensor."""
    # Mock tokenizer if None (ASCII)
    all_ids = []
    for item in data_list:
        text = item.get("text", "")
        if "question" in item:
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"
        elif "prompt" in item:
            text = f"User: {item['prompt']}\nAssistant: {item['response']}"

        ids = [ord(c) % 256 for c in text] # Mock char-level
        all_ids.extend(ids)

    tensor = torch.tensor(all_ids, dtype=torch.long)
    torch.save(tensor, path)
    print(f"   Saved {len(all_ids)} tokens to {path}")

def download_or_mock(phase_name, ds_id, ds_split, limit=1000):
    print(f"📥 processing {phase_name}: {ds_id}...")
    output_path = os.path.join(DATA_DIR, f"{phase_name}/train.pt")

    data = []
    if HAS_DATASETS:
        try:
            ds = load_dataset(ds_id, split=ds_split, streaming=True)
            count = 0
            for item in ds:
                data.append(item)
                count += 1
                if count >= limit: break
            print(f"   Downloaded {len(data)} items.")
        except Exception as e:
            print(f"   ❌ Download failed ({e}). Falling back to MOCK.")
            HAS_DATASETS_FAIL = True

    if not data:
        print("   Generating MOCK data...")
        for i in range(limit):
            if "gsm8k" in ds_id:
                data.append({"question": f"What is {i}+{i}?", "answer": f"{i+i}"})
            elif "oasst" in ds_id:
                data.append({"prompt": f"Tell me a joke {i}", "response": f"Joke {i}"})
            else:
                data.append({"text": f"This is synthetic text sample number {i} for training phase {phase_name}."})

    save_to_bin(data, output_path)

def main():
    print("🚀 Starting Industrial Data Pipeline...")

    # Phase 0: Sanity (WikiText-2)
    download_or_mock("phase_0", "Salesforce/wikitext", "train", limit=5000)

    # Phase 1: Fluency (WikiText-103) - Mocked mainly
    download_or_mock("phase_1", "Salesforce/wikitext", "train", limit=10000)

    # Phase 2: Reasoning (GSM8K, ARC)
    download_or_mock("phase_2", "openai/gsm8k", "train", limit=2000)

    # Phase 3: Chat (OASST1)
    download_or_mock("phase_3", "h2oai/openassistant_oasst1_h2ogpt", "train", limit=2000)

    print("✅ Data Pipeline Complete.")

if __name__ == "__main__":
    main()
