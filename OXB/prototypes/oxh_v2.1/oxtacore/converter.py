import json
import struct
import tiktoken

class OXBConverter:
    def __init__(self, tokenizer_name="cl100k_base"):
        self.enc = tiktoken.get_encoding(tokenizer_name)

    def convert_jsonl(self, jsonl_path, output_prefix, shard_size=100000, text_key="text", score_key="score"):
        shard_count = 0
        record_count = 0

        def open_shard(c):
            p = f"{output_prefix}_part_{c:03d}.oxh"
            f_d = open(p, 'wb')
            f_i = open(p + ".idx", 'wb')
            f_d.write(struct.pack('<4sHH', b'OXH2', 2, 0)) # Header v2
            return f_d, f_i, p

        f_data, f_idx, current_path = open_shard(shard_count)

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if record_count > 0 and record_count % shard_size == 0:
                    f_data.close(); f_idx.close()
                    shard_count += 1
                    f_data, f_idx, _ = open_shard(shard_count)

                obj = json.loads(line)
                tokens = self.enc.encode(obj.get(text_key, ""))
                score = float(obj.get(score_key, 1.0))

                f_idx.write(struct.pack('<Q', f_data.tell())) # Indexação

                # Bit-Packing Híbrido
                buf = bytearray()
                for t in tokens:
                    if t < 65535:
                        buf.extend(struct.pack('<H', t))
                    else:
                        buf.extend(struct.pack('<H', 65535))
                        buf.extend(struct.pack('<I', t))

                # Gravação: [Score f32][Size u32][Payload]
                f_data.write(struct.pack('<f', score))
                f_data.write(struct.pack('<I', len(buf)))
                f_data.write(buf)
                record_count += 1

        f_data.close(); f_idx.close()
        print(f"Finalizado: {record_count} amostras convertidas.")
