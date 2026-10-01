"""trainer_main.py — entrypoint of the standalone Windows .exe.

This script is the only Python source that gets compiled into the
PyInstaller bundle.  All references to the underlying architecture
(model family, internal cherry-picks, training profile names) are
deliberately stripped from text strings the user could see — terminology
is replaced with neutral names so the bundle reveals minimal intent.

Folder layout assumed at runtime (relative to .exe location):

    OxtaTrainer/
    ├── OxtaTrainer.exe                ← this script, frozen
    ├── _internal/                     ← PyInstaller stash (Python DLLs, .pyd)
    │   └── nsos_ext.cp311-win_amd64.pyd
    ├── data/
    │   ├── bundle/                    ← tokenizer + curriculum
    │   ├── datasets/                  ← raw corpora
    │   ├── config/runtime.json        ← all hyperparams (loaded at start)
    │   └── manifest.json              ← integrity check
    ├── output/                        ← created at runtime
    │   ├── checkpoints/               ← intermediate state every N steps
    │   ├── logs/session.log           ← human-readable progress
    │   └── final_model.bin            ← what the operator returns to us
    └── README.txt                     ← bundled separately

The user runs OxtaTrainer.exe.  Nothing else.
"""
from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Path discovery ──────────────────────────────────────────────────────────
# When frozen by PyInstaller, sys.executable points to the .exe.  In dev mode
# (running the .py directly) it points to python.exe — fall back to the
# script's own directory then.
def discover_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


ROOT = discover_root()
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
LOGS_DIR = OUTPUT_DIR / "logs"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"

# Ensure output tree exists before we try to log anything.
for p in (OUTPUT_DIR, LOGS_DIR, CHECKPOINTS_DIR):
    p.mkdir(parents=True, exist_ok=True)


# ── Lightweight logger that writes to both stdout and session.log ──────────
# We don't import the stdlib logging module to keep startup fast and to
# minimize the surface visible in the frozen bundle.
SESSION_LOG = LOGS_DIR / "session.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with SESSION_LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as exc:
        # stdout remains available, but the loss of durable logging is visible.
        print(f"[logger] session.log write failed: {exc}", file=sys.stderr)


def banner(title: str) -> None:
    log("=" * 70)
    log(f"  {title}")
    log("=" * 70)


# ── Pre-flight checks ──────────────────────────────────────────────────────
def check_environment() -> None:
    banner("Verificando ambiente")
    log(f"  platform:  {platform.system()} {platform.release()} {platform.machine()}")
    log(f"  python:    {sys.version.split()[0]}")
    log(f"  cwd:       {Path.cwd()}")
    log(f"  data dir:  {DATA_DIR}")
    log(f"  output:    {OUTPUT_DIR}")

    if not DATA_DIR.exists():
        log(f"ERRO: pasta 'data/' nao encontrada em {DATA_DIR}")
        log("  Esse executavel precisa rodar do diretorio que contem 'data/'.")
        sys.exit(2)

    needed = ["bundle/tokenizer_8192.ox3", "config/runtime.json", "manifest.json"]
    for rel in needed:
        path = DATA_DIR / rel
        if not path.exists():
            log(f"ERRO: arquivo necessario ausente: data/{rel}")
            sys.exit(2)
    log("  ok")


# Compute-capability targets baked into the .pyd at build time.  The kernels
# bundled here were compiled for sm_75 (Turing -- RTX 2080 / 2080 Ti / T4 /
# Quadro RTX, and Tesla T4).  Newer cards (Ampere sm_8x, Ada sm_89, Hopper
# sm_9x) can JIT from the embedded PTX, but older cards (Pascal sm_61 like
# GTX 1050/1060/1070/1080, Maxwell sm_5x, Kepler sm_3x) cannot run sm_75
# kernels and can't be JIT'd backwards either.  Better to refuse at startup
# than to die mid-training with "no kernel image" errors.
SUPPORTED_COMPUTE_CAPS_MIN = (7, 5)  # sm_75 inclusive
SUPPORTED_GPU_NAMES_FALLBACK = (
    # Match these substrings if compute_cap query fails.
    "RTX 20", "RTX 30", "RTX 40", "RTX A", "T4", "T1000", "Quadro RTX",
    "A100", "A40", "A10", "A30", "L4", "L40", "H100", "H200",
)


def _query_compute_cap(name_hint: str) -> Optional[tuple]:
    """Ask nvidia-smi for compute capability.  Returns (major, minor) or None."""
    import subprocess
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        first = r.stdout.strip().split("\n")[0].strip()
        if "." in first:
            major, minor = first.split(".", 1)
            return (int(major), int(minor))
    except Exception as exc:
        log(f"  aviso: compute capability query failed: {exc}")
    return None


def check_gpu() -> Dict[str, str]:
    banner("Verificando GPU")
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        info = {
            "name": parts[0] if len(parts) > 0 else "?",
            "driver": parts[1] if len(parts) > 1 else "?",
            "memory_mb": parts[2] if len(parts) > 2 else "?",
            "free_mb": parts[3] if len(parts) > 3 else "?",
            "used_mb": parts[4] if len(parts) > 4 else "?",
        }

        def _gb(mb: str) -> str:
            try:
                return f"{float(mb) / 1024:.1f} GB"
            except (TypeError, ValueError):
                return "? GB"

        log(f"  GPU:        {info['name']}")
        log(f"  driver:     {info['driver']}")
        log(f"  VRAM total: {info['memory_mb']} MB ({_gb(info['memory_mb'])})  <- VRAM dedicada real (a que a CUDA usa)")
        log(f"  VRAM livre: {info['free_mb']} MB ({_gb(info['free_mb'])})")
        log(f"  VRAM usada: {info['used_mb']} MB")
        log(f"  nota: os 'GB' nas Propriedades do Windows = dedicada + compartilhada (RAM);")
        log(f"        o numero que importa para o treino e o 'VRAM total' acima.")

        # Compute capability gate -- refuse to start if too old.
        cc = _query_compute_cap(info["name"])
        if cc is not None:
            log(f"  compute:   sm_{cc[0]}{cc[1]}")
            min_major, min_minor = SUPPORTED_COMPUTE_CAPS_MIN
            if (cc[0], cc[1]) < (min_major, min_minor):
                log(f"")
                log(f"ERRO: GPU muito antiga para esta build.")
                log(f"  Detectado:  sm_{cc[0]}{cc[1]} ({info['name']})")
                log(f"  Necessario: sm_{min_major}{min_minor} ou mais novo")
                log(f"  ")
                log(f"  Esta build foi compilada para arquitetura Turing (RTX 20-series).")
                log(f"  Cartoes mais antigos (GTX 10xx Pascal, GTX 9xx Maxwell) nao rodam.")
                log(f"  Cartoes mais novos (RTX 30xx, 40xx) devem funcionar via JIT.")
                sys.exit(3)
        else:
            # Fallback: heuristic match on GPU name
            name_upper = info["name"].upper()
            matches = any(p.upper() in name_upper for p in SUPPORTED_GPU_NAMES_FALLBACK)
            if not matches:
                log(f"")
                log(f"AVISO: nao consegui verificar compute_cap; nome da GPU "
                    f"({info['name']!r}) nao esta na lista conhecida.")
                log(f"  Vou tentar rodar mesmo assim, mas se houver erro 'no kernel "
                    f"image' contacte o operador.")
        return info
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        log(f"ERRO: GPU NVIDIA nao encontrada. nvidia-smi falhou: {exc}")
        log("  Verifique driver NVIDIA + CUDA Toolkit instalados.")
        sys.exit(3)


# ── Load config from data/config/runtime.json ──────────────────────────────
def load_runtime_config() -> Dict[str, Any]:
    config_path = DATA_DIR / "config" / "runtime.json"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    log(f"  config carregado de {config_path.relative_to(ROOT)}")
    return cfg


# ── Cherry-pick env vars (set before loading the native module) ────────────
def apply_runtime_env() -> None:
    # These flags activate the optimized kernel paths inside the native
    # module.  They must be set BEFORE the .pyd is imported because the
    # module captures them on initialization.
    os.environ.setdefault("NSOS_MIXED_PRECISION", "bf16")
    os.environ.setdefault("NSOS_USE_LUT_SIMD", "1")
    # Force release of intermediate FP32 weights after pack load to save VRAM
    # on the friend's 11GB card.  Inference would still work but slightly
    # less precise — for training, we keep FP32 master so this is unset.


# ── Native module load ─────────────────────────────────────────────────────
def _register_dll_search_paths() -> None:
    """Python 3.8+ on Windows restricts DLL lookup for native extensions.
    The bundled CUDA runtime DLLs sit beside the .pyd inside _internal/, so
    register that directory (and the bundle root, just in case) so the
    loader can resolve cudart64_*.dll / cublas64_*.dll without leaning on
    PATH or requiring CUDA Toolkit on the user's PC."""
    if not hasattr(os, "add_dll_directory"):
        return  # not Windows; nothing to do
    candidates = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        candidates += [exe_dir, exe_dir / "_internal"]
    # Also expose any CUDA Toolkit install present on the machine as a
    # last-resort fallback (useful when running this script directly during
    # dev, not bundled).
    for env_var in ("CUDA_PATH", "CUDA_HOME"):
        val = os.environ.get(env_var)
        if val:
            candidates.append(Path(val) / "bin")
    seen = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not resolved.is_dir() or resolved in seen:
            continue
        seen.add(resolved)
        try:
            os.add_dll_directory(str(resolved))
        except (OSError, FileNotFoundError) as exc:
            log(f"  aviso: DLL directory registration failed for {resolved}: {exc}")


def load_engine_module():
    banner("Carregando engine nativo (init CUDA ~30-90s na primeira vez)")
    _register_dll_search_paths()
    try:
        import nsos_ext  # noqa
        log(f"  engine carregado de {nsos_ext.__file__}")
        return nsos_ext
    except ImportError as exc:
        log(f"ERRO: falha ao importar engine nativo: {exc}")
        log("  Verifique se nsos_ext.pyd esta presente no bundle.")
        sys.exit(4)


# ── Dataset enumeration (just scans data/datasets/ for jsonl files) ────────
def enumerate_training_data() -> List[Path]:
    datasets_dir = DATA_DIR / "datasets"
    if not datasets_dir.exists():
        log(f"ERRO: data/datasets/ nao encontrado")
        sys.exit(2)

    # Accept .jsonl (raw) or .jsonl.zst (compressed) — we'll handle both.
    files: List[Path] = []
    for ext in ("*.jsonl", "*.jsonl.zst"):
        files.extend(sorted(datasets_dir.rglob(ext)))

    log(f"  encontrados {len(files)} arquivos de treino em {datasets_dir.relative_to(ROOT)}")
    if not files:
        log(f"ERRO: nenhum dataset encontrado em {datasets_dir}")
        sys.exit(2)

    total_bytes = sum(f.stat().st_size for f in files)
    log(f"  tamanho total bruto: {total_bytes / 1024**3:.1f} GB")
    return files


# ── Iterate documents from a (.jsonl | .jsonl.zst) file ────────────────────
def iter_documents(file_path: Path):
    """Yield text strings from a JSONL file, decompressing zst if needed."""
    if file_path.suffix == ".zst":
        try:
            import zstandard as zstd  # type: ignore
        except ImportError:
            log("ERRO: zstandard nao disponivel; sem suporte a .jsonl.zst")
            return
        with file_path.open("rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                buf = b""
                while True:
                    chunk = reader.read(1 << 16)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, _, buf = buf.partition(b"\n")
                        try:
                            row = json.loads(line.decode("utf-8"))
                            text = row.get("text", "")
                            if text:
                                yield text
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue
    else:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    text = row.get("text", "")
                    if text:
                        yield text
                except json.JSONDecodeError:
                    continue


# ── Streaming document provider (bounded RAM) ──────────────────────────────
def stream_documents(files, min_len=200, buffer_size=8192, seed=1337):
    """Yield documents lazily through a bounded shuffle buffer, cycling epochs
    forever.  Keeps RAM ~constant (a few thousand docs) instead of loading the
    whole dataset into a Python list -- which was pinning the box at ~98% RAM
    and freezing it / killing AnyDesk."""
    import random as _random
    rng = _random.Random(seed)
    while True:                       # cycle epochs forever
        order = list(files)
        rng.shuffle(order)
        buf = []
        for fp in order:
            for text in iter_documents(fp):
                if len(text) >= min_len:
                    buf.append(text)
                    if len(buf) >= buffer_size:
                        rng.shuffle(buf)
                        for d in buf:
                            yield d
                        buf = []
        if buf:
            rng.shuffle(buf)
            for d in buf:
                yield d


def log_gpu_status(tag=""):
    """Log GPU utilization + VRAM via nvidia-smi, so it's obvious the GPU is hot."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            util, used, total = (p.strip() for p in out.stdout.strip().splitlines()[0].split(","))
            log(f"  [GPU{(' ' + tag) if tag else ''}] uso={util}%  VRAM={used}/{total} MB")
    except Exception as exc:
        log(f"  aviso: GPU status query failed: {exc}")


def _dump_gpu_processes():
    """List the processes currently holding VRAM (to spot zombie runs)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        body = out.stdout.strip()
        if body:
            for line in body.splitlines():
                log(f"     VRAM em uso por: {line.strip()}")
        else:
            log("     (nenhum processo segurando a GPU agora)")
    except Exception as exc:
        log(f"  aviso: GPU process query failed: {exc}")


def log_ram_status(tag=""):
    """Log system RAM load via the Win32 API (no extra dependency)."""
    try:
        import ctypes
        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExt", ctypes.c_ulonglong)]
        s = _MS(); s.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        used = (s.ullTotalPhys - s.ullAvailPhys) / 1024**3
        total = s.ullTotalPhys / 1024**3
        log(f"  [RAM{(' ' + tag) if tag else ''}] uso={s.dwMemoryLoad}%  ({used:.1f}/{total:.1f} GB)")
    except Exception as exc:
        log(f"  aviso: RAM status query failed: {exc}")


# ── Resume support ─────────────────────────────────────────────────────────
def find_latest_checkpoint() -> Optional[Path]:
    """Return the most recently modified checkpoint .bin, if any."""
    candidates = list(CHECKPOINTS_DIR.glob("*.bin"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ── Main training loop ─────────────────────────────────────────────────────
def run_training(cfg: Dict[str, Any], engine_module) -> int:
    banner("Inicializando modelo")

    # Build model config from runtime.json (no profile names in source)
    mcfg = engine_module.ModelConfig()
    arch = cfg["model"]
    mcfg.num_layers          = arch["num_layers"]
    mcfg.d_model             = arch["d_model"]
    mcfg.vocab_size          = arch["vocab_size"]
    mcfg.max_context_tokens  = arch["max_context_tokens"]
    mcfg.n_heads             = arch["n_heads"]
    mcfg.n_kv_heads          = arch["n_kv_heads"]
    mcfg.sliding_window      = arch["sliding_window"]
    mcfg.attention_period    = arch["attention_period"]
    mcfg.attention_slot      = arch["attention_slot"]
    mcfg.use_moe             = arch["use_moe"]
    mcfg.num_experts         = arch["num_experts"]
    mcfg.num_experts_per_token = arch["num_experts_per_token"]
    mcfg.moe_period          = arch["moe_period"]
    mcfg.moe_slot            = arch["moe_slot"]
    # Cherry-pick #4: K·m invariant tuning (m=0 -> default dm*4)
    if "moe_expert_hidden_dim" in arch:
        mcfg.moe_expert_hidden_dim = arch["moe_expert_hidden_dim"]
    mcfg.use_ttt             = False
    mcfg.use_gradient_checkpointing = arch["use_gradient_checkpointing"]
    mcfg.dropout             = arch.get("dropout", 0.0)
    mcfg.use_cuda            = True
    mcfg.default_batch_size  = arch["batch_size"]
    mcfg.use_exact_attention_training = arch["use_exact_attention_training"]
    mcfg.use_flash_attn      = arch.get("use_flash_attn", False)

    engine = engine_module.InferenceEngine()

    # Confirm the GPU is free + has headroom BEFORE we try to allocate the model.
    log_gpu_status("antes de init")
    log_ram_status("antes de init")

    # Resume or fresh init
    resume_path = find_latest_checkpoint()
    try:
        if resume_path is not None:
            log(f"  retomando de {resume_path.relative_to(ROOT)}")
            ok = engine.load_model(str(resume_path), mcfg)
        else:
            log(f"  init fresh com {mcfg.num_layers} layers, d={mcfg.d_model}")
            ok = engine.load_model("", mcfg)
    except Exception as exc:
        log(f"ERRO: load_model lancou {type(exc).__name__}: {exc}")
        log("  -> em geral e VRAM ocupada por um processo-zumbi de um run travado.")
        log("  -> rode 'nvidia-smi', mate processos python que ainda seguram a GPU")
        log("     (ou reinicie a maquina) e tente de novo. Estado atual da GPU:")
        log_gpu_status("na falha")
        _dump_gpu_processes()
        return 1

    if not ok:
        log("ERRO: engine.load_model retornou False")
        log_gpu_status("na falha (ok=False)")
        _dump_gpu_processes()
        return 1

    log(f"  memory after init: {engine.get_memory_usage()}")
    log_gpu_status("apos carregar modelo")
    log_ram_status("apos carregar modelo")

    # ── Datasets: STREAM from disk. Do NOT load every doc into a Python list --
    # that pinned the box at ~98% RAM and froze the machine / killed AnyDesk.
    # A bounded shuffle buffer keeps RAM roughly constant regardless of dataset
    # size, and the generator cycles epochs forever.
    banner("Preparando datasets (streaming, RAM controlada)")
    files = enumerate_training_data()
    seed = int(cfg.get("training", {}).get("seed", 1337))
    shuffle_buffer = int(cfg.get("training", {}).get("shuffle_buffer_docs", 8192))
    log(f"  modo: streaming on-demand (buffer {shuffle_buffer} docs)")
    log_ram_status("apos enumerar datasets")
    doc_stream = stream_documents(files, min_len=200, buffer_size=shuffle_buffer, seed=seed)

    # ── Training loop
    banner("Iniciando treino")
    training_cfg = cfg["training"]
    target_steps = training_cfg["target_steps"]
    checkpoint_every = training_cfg["checkpoint_every_steps"]
    log_every = training_cfg.get("log_every_steps", 25)
    max_doc_chars = training_cfg.get("max_doc_chars", 4096)

    log(f"  target steps:        {target_steps}")
    log(f"  checkpoint every:    {checkpoint_every}")
    log(f"  log every:           {log_every}")
    log(f"  max doc chars:       {max_doc_chars}")

    losses: List[float] = []
    t_start = time.time()

    log_gpu_status("antes do treino")   # confirm the GPU is loaded and about to be used
    log_ram_status("antes do treino")

    for step in range(1, target_steps + 1):
        t0 = time.time()
        doc = next(doc_stream)

        try:
            result = engine.train_text(doc[:max_doc_chars])
            loss = float(result) if isinstance(result, (int, float)) \
                   else float(result.get("loss", float("nan")))
        except Exception as exc:
            log(f"step {step}: train_text failed: {type(exc).__name__}: {exc}")
            continue

        losses.append(loss)
        step_dt = time.time() - t0

        if step % log_every == 0 or step == 1:
            recent = losses[-log_every:]
            avg = sum(recent) / len(recent)
            elapsed = time.time() - t_start
            eta_h = (target_steps - step) * (elapsed / step) / 3600
            log(f"  step={step:>5d}/{target_steps}  loss={avg:.3f}  "
                f"step_t={step_dt:.2f}s  eta={eta_h:.1f}h")
            log_gpu_status()   # uso=% confirms the GPU is hot; VRAM tracks the budget
            log_ram_status()   # should stay flat/low now that we stream

        if step % checkpoint_every == 0:
            ckpt = CHECKPOINTS_DIR / f"step_{step:08d}.bin"
            try:
                engine.save_checkpoint(str(ckpt))
                log(f"  checkpoint salvo: {ckpt.name}")
            except Exception as exc:
                log(f"WARNING: save_checkpoint falhou: {exc}")

    total_h = (time.time() - t_start) / 3600
    banner(f"Treino completo em {total_h:.1f}h")

    # Final pack
    final_pack_dir = OUTPUT_DIR / "final_pack"
    final_pack_dir.mkdir(exist_ok=True)
    try:
        engine.save_model_pack(str(final_pack_dir))
        log(f"  pack final salvo em {final_pack_dir.relative_to(ROOT)}")
    except Exception as exc:
        log(f"WARNING: save_model_pack falhou: {exc}")
        # Fallback to single checkpoint
        try:
            engine.save_checkpoint(str(OUTPUT_DIR / "final_model.bin"))
            log(f"  fallback: final_model.bin salvo em output/")
        except Exception as exc2:
            log(f"ERRO: save_checkpoint final tambem falhou: {exc2}")
            return 1

    # Summary
    if losses:
        log(f"\n  loss inicial:   {sum(losses[:10])/min(10, len(losses)):.3f}")
        log(f"  loss final:     {sum(losses[-25:])/min(25, len(losses)):.3f}")
        log(f"  loss min:       {min(losses):.3f}")
        log(f"  steps:          {len(losses)}/{target_steps}")
    log("")
    log("Tarefa concluida. Envie o conteudo da pasta 'output/' para o operador.")
    return 0


# ── Signal handling so Ctrl+C exits cleanly with last checkpoint preserved ─
def install_signal_handlers() -> None:
    def handler(signum, frame):
        log(f"  sinal {signum} recebido, encerrando.  Checkpoint preservado.")
        sys.exit(130)
    try:
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    except (ValueError, AttributeError) as exc:
        log(f"  aviso: signal handlers unavailable: {exc}")


def main() -> int:
    install_signal_handlers()
    banner("OxtaTrainer v1 — standalone")
    log(f"  iniciado em {time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        check_environment()
        check_gpu()
        cfg = load_runtime_config()
        apply_runtime_env()
        engine = load_engine_module()
        return run_training(cfg, engine)
    except KeyboardInterrupt:
        log("  interrompido pelo operador")
        return 130
    except Exception as exc:
        log(f"ERRO fatal: {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        return 99


if __name__ == "__main__":
    sys.exit(main())
