# Configuration for Benchmarks
NUM_SAMPLES = 5000
BATCH_SIZE = 64
NUM_WORKERS = 2 # Sandbox limit usually
MAX_LEN = 128
VOCAB_SIZE = 100277

# File Paths
DATA_DIR = "bench_data"
JSONL_FILE = f"{DATA_DIR}/data.jsonl"
TOON_FILE = f"{DATA_DIR}/data.toon"
OXH_PREFIX = f"{DATA_DIR}/data_oxh"
