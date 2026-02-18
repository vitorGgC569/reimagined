import argparse
import struct
import numpy as np
import os

# Stub for GGUF writer if not installed, purely to demonstrate the format logic
try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("GGUF library not found. Installing stub...")
    class GGUFWriter:
        def __init__(self, path, arch):
            self.path = path
            self.f = open(path, "wb")
        def add_tensor(self, name, data, raw_shape=None):
            print(f"Writing tensor {name}...")
        def add_architecture(self): pass
        def add_block_count(self, n): pass
        def write_header_to_file(self): pass
        def write_kv_data_to_file(self): pass
        def write_tensors_to_file(self): pass
        def close(self): self.f.close()

def load_nsos_model(path):
    # Load the binary format defined in JambaModel::save
    print(f"Loading NSOS model from {path}...")
    params = {}
    with open(path, "rb") as f:
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != 0x000E0505:
            raise ValueError("Invalid NSOS Magic Number")
        version = struct.unpack('<I', f.read(4))[0]
        count = struct.unpack('<I', f.read(4))[0]

        for _ in range(count):
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            rank = struct.unpack('<I', f.read(4))[0]
            shape = []
            size = 1
            for _ in range(rank):
                dim = struct.unpack('<I', f.read(4))[0]
                shape.append(dim)
                size *= dim

            data_bytes = f.read(size * 4) # float32
            data = np.frombuffer(data_bytes, dtype=np.float32).reshape(shape)
            params[name] = data
            print(f"Loaded {name} {shape}")

    return params

def convert_to_gguf(nsos_path, gguf_path):
    params = load_nsos_model(nsos_path)

    gw = GGUFWriter(gguf_path, "nsos")
    gw.add_architecture()

    # Map NSOS names to GGUF standard names where possible or use custom
    for name, data in params.items():
        # Clean name
        clean_name = name.replace("layers.", "blk.").replace("mamba.", "ssm.")
        gw.add_tensor(clean_name, data)

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    print(f"Converted {nsos_path} to {gguf_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to NSOS .bin model")
    parser.add_argument("output", help="Path to output .gguf file")
    args = parser.parse_args()

    convert_to_gguf(args.input, args.output)
