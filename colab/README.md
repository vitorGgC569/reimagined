# NSOS v11 â€” Colab Federated Training

This directory contains the infrastructure for running NSOS training
on Google Colab while keeping authoritative checkpoints on your local
machine, mediated by Google Drive (5TB on the user's Google AI Pro plan).

## Current validated benchmark

For the July 2026 Tesla T4 comparison of NSOS, reference Jamba dense,
reference Jamba MoE and OxtaMem, start with:

- [`bench_expanded_validation_t4_2026-07-22.ipynb`](bench_expanded_validation_t4_2026-07-22.ipynb) â€” executed, clone-contained notebook;
- [`../docs/benchmarks/jamba_vs_nsos_oxtamem_t4_2026-07-22.md`](../docs/benchmarks/jamba_vs_nsos_oxtamem_t4_2026-07-22.md) â€” protocol, results, limitations and verdict;
- [`../docs/benchmarks/results/expanded_validation_t4_2026-07-22.json`](../docs/benchmarks/results/expanded_validation_t4_2026-07-22.json) â€” raw results.

The source notebook can be opened directly in
[Colab from GitHub](https://colab.research.google.com/github/vitorGgC569/reimagined/blob/nsos-gpu-phases12/colab/bench_expanded_validation_t4_2026-07-22.ipynb).
Because the repository is private, Colab must be authorized to read it.
No secret is stored in the notebook or repository.

The federated v11 instructions below are a separate, older workflow. They are
not required to reproduce the July 2026 benchmark.

## Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Colab (Linux, T4/A100/L4, ephemeral 12-24h sessions)               â”‚
â”‚  â”€ git clone repo                                                   â”‚
â”‚  â”€ build nsos_ext.so (cached in Drive per commit/arch)              â”‚
â”‚  â”€ run train_curriculum.py --run-dir /content/drive/.../runs/...    â”‚
â”‚  â”€ checkpoints land directly in Drive every 100 steps               â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                              â†•
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Google Drive (5TB, cloud storage)                                  â”‚
â”‚  /MyDrive/nsos_v11/                                                 â”‚
â”‚  â”œâ”€â”€ _bootstrap/        â€” cached .so files keyed by git SHA + arch  â”‚
â”‚  â”œâ”€â”€ _datasets_raw/     â€” HuggingFace dataset dumps                 â”‚
â”‚  â”œâ”€â”€ _datasets_processed/ â€” built distillation bundles              â”‚
â”‚  â””â”€â”€ runs/<run_name>/   â€” checkpoints + metrics per run             â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                              â†•
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Local Windows machine (your PC, authoritative storage)             â”‚
â”‚  â”€ python OXN/nsos/scripts/drive_sync.py --interval 300             â”‚
â”‚    polls Drive every 5 min, mirrors new files locally               â”‚
â”‚  â”€ chat / inference uses local checkpoints (offline-capable)        â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

## Files in this directory

| File | Purpose |
|------|---------|
| `bench_expanded_validation_t4_2026-07-22.ipynb` | Reproducible T4 benchmark: native builds, multi-seed MQAR, bAbI QA1, OxtaMem and extrapolation. |
| `train_v11.ipynb` | The Colab notebook you open in your browser. Six cells: mount Drive â†’ clone repo â†’ bootstrap â†’ fetch datasets (one-time) â†’ train â†’ verify. |
| `colab_bootstrap.py` | Heavy-lifting Python module imported by the notebook. Detects GPU, builds/caches .so, extracts datasets. |
| `README.md` | This file. |

## Files outside this directory (referenced)

| Path | Purpose |
|------|---------|
| `OXN/nsos/scripts/train_curriculum.py` | Trainer. Profiles `hybrid_v11_colab_t4`, `hybrid_v11_colab_a100`, `hybrid_v11_colab_l4` added for the federated setup. |
| `OXN/nsos/scripts/gpu_arch_detect.py` | Detects host GPU compute capability for CMake. Used by build scripts and the bootstrap. |
| `OXN/nsos/scripts/fetch_real_datasets.py` | Fetches the 17 datasets (11 original + 6 v11 expansion: Cosmopedia, TinyStories, SmolTalk, The Stack, C4, SQuAD v2). Supports `--scale-factor` to multiply target row counts. |
| `OXN/nsos/scripts/drive_sync.py` | Local-side daemon. Polls Drive Desktop mount (default) or rclone, mirrors new files into `OXN/nsos/scripts/live_distill_v11_colab/`. |

## First-time setup (~2-3 hours, one time)

### Step 1 â€” Get the repo on GitHub

If you don't have it on GitHub yet, push the local repo:
```powershell
cd C:\Users\Oxta\Desktop\reimagined-main
git remote add origin https://github.com/<YOUR_USERNAME>/reimagined-main.git
git push -u origin main
```

If the repo should be private, set up a GitHub Personal Access Token
and add it to Colab's Secrets panel as `GH_TOKEN`.

### Step 2 â€” Install Google Drive Desktop on Windows

Download from <https://www.google.com/drive/download/>. After install
your Drive is accessible at `G:\My Drive\` (or similar letter). The
`drive_sync.py` script auto-detects it.

If you prefer rclone instead, install it and run `rclone config` to
set up a `gdrive` remote, then set `NSOS_RCLONE_REMOTE=gdrive:nsos_v11`
in your environment.

### Step 3 â€” Open the notebook in Colab

1. Go to <https://colab.research.google.com/>
2. File â†’ Open notebook â†’ GitHub tab â†’ enter your repo URL â†’ select `colab/train_v11.ipynb`
3. Runtime â†’ Change runtime type â†’ GPU â†’ T4 (or A100/L4 if you have compute units)
4. Edit cell 2 (`step2_code`): replace `<YOUR_GIT_REMOTE>` with your GitHub username
5. Run all cells (Runtime â†’ Run all)

First session takes ~30 min for setup (build .so, fetch datasets) plus the actual training.
Subsequent sessions skip setup (everything cached in Drive).

### Step 4 â€” Start the local sync daemon

In a separate terminal on your local Windows machine, leave this running:
```powershell
cd C:\Users\Oxta\Desktop\reimagined-main\.claude\worktrees\clever-roentgen-ba007c\OXN\nsos\scripts
python drive_sync.py --interval 300
```

Every 5 minutes it polls Drive Desktop and mirrors any new checkpoints
to `OXN/nsos/scripts/live_distill_v11_colab/`. You can then use those
locally with `chat_v10_ui.py` (pointing it at the v11 directory).

## What runs where

| Action | Where it runs | Why |
|---|---|---|
| Code editing | Local Windows | Owned environment, fast IDE |
| Git commit + push | Local Windows | Your authorship |
| Build `nsos_ext` for Colab | Colab (one-time per commit, cached) | Colab Linux + sm_75/80 toolchain |
| Build `nsos_ext` for local | Local Windows | sm_61 Pascal toolchain |
| Dataset fetch | Colab (Google's bandwidth) | ~50GB download, much faster than your home internet |
| Training | Colab GPU | T4 = 12-15Ã— faster than your 1050 Ti |
| Inference / chat | Local Windows | Sovereignty, offline use |
| Checkpoint storage | Drive (authoritative cloud) + local mirror | Survives Colab session death, doesn't survive Drive deletion alone |

## Storage budget (per run)

| Item | Size | Where |
|------|------|-------|
| Built `nsos_ext.so` (per commit/arch) | ~5 MB | Drive `/_bootstrap/` |
| Raw datasets (v11 expanded) | ~30-50 GB | Drive `/_datasets_raw/` (one-time) |
| Processed bundle (`distillation_bundle_v11.zip`) | ~2-5 GB | Drive `/_datasets_processed/` (one-time) |
| Per-checkpoint (phase end) | ~390 MB each | Drive `/runs/<name>/` |
| Full run history (~15 checkpoints) | ~6 GB | Drive `/runs/<name>/` |

Your 5TB Drive can hold ~800 full runs without issue.

## Cost estimates (USD)

| Plan | GPU available | v11 training time (800M tokens) | Extra cost |
|------|--------------|-------------------------------|------------|
| Colab Free | T4 only (best-effort) | ~13 sessions Ã— 12h = ~2-3 weeks calendar | $0 |
| Colab Pro ($10/mo) | T4 priority + L4 sometimes | ~2-3 sessions Ã— 24h = ~1 week | $10 |
| Colab Pro+ ($50/mo) | A100 priority, 24h continuous | ~1 session of 4-5h | $50 |
| Pay-as-you-go ($9.99 per 100 units) | A100 if you wait | 1 session of 5h consumes ~75 units | $7.50 effective |

For v11 specifically, **pay-as-you-go ($7.50 for one A100 session)** is likely the best value.

## Troubleshooting

**"Build fails â€” pybind11 not found"**  
Bootstrap runs `pip install pybind11>=2.10`. If it fails, try `!apt install python3-pybind11 -y` first.

**"GPU not allocated even though I selected T4"**  
Colab Free sometimes can't give you a GPU at peak hours. Disconnect and retry in 10 min, or pay $9.99 for compute units.

**"My local sync daemon doesn't see new checkpoints"**  
Drive Desktop syncs from cloud â†’ local on its own schedule (usually <2 min). Check Drive Desktop's icon in the system tray; it shows last sync time. If stuck, restart Drive Desktop.

**"Training crashes mid-session with OOM"**  
For T4, the `hybrid_v11_colab_t4` profile (batch=32) is tuned for 16GB. If it OOMs, edit the profile in `train_curriculum.py` to `batch_size=24` or `28` and re-run. Drive checkpoint = no progress lost.

