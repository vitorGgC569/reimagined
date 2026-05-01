import json
import struct
import tiktoken
import numpy as np
from .learned_index import train_and_save_index

class OXBConverterV3:
    def __init__(self, tokenizer_name="cl100k_base"):
        self.enc = tiktoken.get_encoding(tokenizer_name)

    def convert_jsonl(self, jsonl_path, output_prefix, shard_size=100000, text_key="text", score_key="score"):
        shard_count = 0
        record_count = 0
        current_offsets = [] # Keep offsets in memory for training the index at end of shard

        def open_shard(c):
            p = f"{output_prefix}_part_{c:03d}.ox3"
            f_d = open(p, 'wb')
            # Note: We do NOT open a .idx file here anymore. We will save .lin at the end.
            f_d.write(struct.pack('<4sHH', b'OXH3', 3, 0)) # Header v3
            return f_d, p

        f_data, current_path = open_shard(shard_count)
        current_offsets = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if record_count > 0 and record_count % shard_size == 0:
                    # Close current shard
                    f_data.close()
                    # Train and Save Learned Index for this shard
                    train_and_save_index(current_offsets, current_path + ".lin")

                    shard_count += 1
                    f_data, current_path = open_shard(shard_count)
                    current_offsets = []

                # Store current offset before writing
                current_offsets.append(f_data.tell())

                obj = json.loads(line)
                tokens = self.enc.encode(obj.get(text_key, ""))
                score = float(obj.get(score_key, 1.0))

                # --- V3 Delta Encoding ---
                buf = bytearray()

                # 1. Write Token Count (uint32)
                buf.extend(struct.pack('<I', len(tokens)))

                if len(tokens) > 0:
                    # 2. Write Base Token (First one) - absolute
                    last_val = tokens[0]
                    buf.extend(struct.pack('<I', last_val))

                    # 3. Write Deltas
                    for t in tokens[1:]:
                        delta = t - last_val
                        last_val = t

                        # Check if fits in int16
                        if -32768 <= delta <= 32767:
                            if delta == -32768:
                                buf.extend(struct.pack('<H', 0x8000))
                                buf.extend(struct.pack('<i', delta))
                            else:
                                buf.extend(struct.pack('<h', delta))
                        else:
                            buf.extend(struct.pack('<H', 0x8000)) # Escape
                            buf.extend(struct.pack('<i', delta))

                # Gravação: [Magic:Rec][Score f32][Size u32][Payload]
                # We need a Magic Marker for Local Search!
                # If Learned Index prediction is wrong, we scan until we find this marker.
                # Marker: 0xDEADBEEF (4 bytes)
                f_data.write(struct.pack('<I', 0xDEADBEEF))
                f_data.write(struct.pack('<f', score))
                f_data.write(struct.pack('<I', len(buf)))
                f_data.write(buf)
                record_count += 1

        f_data.close()
        # Train final shard index
        if current_offsets:
            train_and_save_index(current_offsets, current_path + ".lin")

        print(f"V3.1 Finalizado: {record_count} amostras convertidas (Delta Encoded + Learned Index).")
