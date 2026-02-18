import argparse
import torch
import struct
import os
import numpy as np

def export_tensor(tensor, name, f):
    """
    Exports a PyTorch tensor to the NSOS binary format.
    Format:
    [Rank (int32)]
    [Dims (Rank * int32)]
    [Data (Size * float32)]
    """
    # Ensure tensor is on CPU and float32
    t = tensor.detach().cpu().float()
    shape = list(t.shape)
    rank = len(shape)

    # Write Header
    f.write(struct.pack('i', rank))
    for dim in shape:
        f.write(struct.pack('i', dim))

    # Write Data efficiently using numpy
    # tofile writes the flat binary data directly
    t.numpy().tofile(f)

    print(f"Exported {name}: {shape}")

def convert_model(model_path, output_path):
    print(f"Loading model from {model_path}...")
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Converting to NSOS binary format at {output_path}...")

    os.makedirs(output_path, exist_ok=True)

    for name, param in model.named_parameters():
        safe_name = name.replace('.', '_')
        file_path = os.path.join(output_path, f"{safe_name}.bin")
        with open(file_path, 'wb') as f:
            export_tensor(param, name, f)

    print("Conversion Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HuggingFace model to NSOS binary format")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID or path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    convert_model(args.model, args.output)
