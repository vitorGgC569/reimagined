# NSOS v11 — Getting Started (Colab)

Step-by-step checklist to start training v11 (80M params, ~1.6B tokens) on Google Colab.

## Pre-flight (one-time, ~30 min)

### ✅ Step 1 — Push the repo to GitHub

The notebook clones from GitHub, so the repo needs to be there.

```powershell
# In Windows PowerShell, at the repo root:
cd C:\Users\Oxta\Desktop\reimagined-main

# If you haven't already, create a GitHub repo at https://github.com/new
# (private is fine; you'll set up a token in step 3)

git remote add origin https://github.com/<YOUR_USERNAME>/reimagined-main.git
git branch -M main
git push -u origin main
```

If `git remote add` errors with "remote origin already exists", that's fine — just `git push -u origin main`.

### ✅ Step 2 — (Optional) Set up GitHub token for private repos

If your repo is **private**:

1. Go to https://github.com/settings/tokens?type=beta
2. Click "Generate new token" → "Fine-grained personal access token"
3. Repository access: select your `reimagined-main` repo
4. Permissions: Contents = Read-only
5. Generate and copy the token (starts with `github_pat_...`)
6. In Colab: click the key icon in the left sidebar → "Add new secret"
7. Name: `GH_TOKEN`, Value: paste your token. Toggle "Notebook access" ON.

If your repo is **public**, skip this step.

### ✅ Step 3 — Install Google Drive Desktop (for local sync later)

Download from https://www.google.com/drive/download/. Install with default settings. After install, your Drive will be at `G:\My Drive\` (or `H:\` if G is taken).

This isn't needed for the Colab side, but you'll want it for the local-sync daemon later.

## Training session (~24 hours total, 2-3 Colab sessions)

### ✅ Step 4 — Open the notebook in Colab

1. Go to https://colab.research.google.com/
2. **File → Open notebook → GitHub tab**
3. URL: `https://github.com/<YOUR_USERNAME>/reimagined-main`
4. Branch: `main`
5. Select `colab/train_v11.ipynb`
6. **Runtime → Change runtime type:**
   - Hardware accelerator: **T4 GPU** (default, free)
   - Version: Latest
   - Click Save

### ✅ Step 5 — Edit cell 2 (GitHub URL)

In the notebook, find the cell that starts with `REPO_URL = ...` and replace `<YOUR_GIT_REMOTE>` with your GitHub username:

```python
REPO_URL = 'https://github.com/<YOUR_USERNAME>/reimagined-main.git'
#                                 ^^^^^^^^^^^^^^^^^ edit this
```

### ✅ Step 6 — Run all cells

**Runtime → Run all** (or Ctrl+F9).

What you'll see, in order:
1. **Cell 2 (mount Drive)**: Asks permission to access Drive — click "Connect to Google Drive" and grant.
2. **Cell 4 (clone repo)**: Should take <30s.
3. **Cell 6 (bootstrap)**: First run takes ~15-20 min (builds nsos_ext.so for T4). Subsequent runs <30s (cache hit).
4. **Cell 8 (fetch datasets)**: First run takes ~1-3h (downloads ~30-50GB to Drive). Subsequent runs <10s.
5. **Cell 10 (training)**: Starts training and runs until session timeout (12h on free) or curriculum completion.

**Total first session: ~14-15h** (15-20 min setup + ~14h training of partial curriculum).

### ✅ Step 7 — On session timeout, just re-open the notebook

When Colab disconnects (after 12h on free tier):

1. Reopen `https://colab.research.google.com/`
2. Open the same `train_v11.ipynb` (Colab remembers it under "Recent")
3. Runtime → Reconnect
4. Runtime → Run all
5. The bootstrap will detect that the .so is cached in Drive (skip build)
6. The trainer will detect the latest checkpoint in Drive and resume from there

Expect 2-3 sessions to complete the full curriculum.

## Monitoring (optional, in parallel)

### Local sync daemon (mirrors Drive → local)

In a separate Windows PowerShell window, leave this running:

```powershell
cd C:\Users\Oxta\Desktop\reimagined-main\.claude\worktrees\clever-roentgen-ba007c\OXN\nsos\scripts
python drive_sync.py --interval 300
```

Every 5 min it pulls new checkpoints from Drive to `OXN/nsos/scripts/live_distill_v11_colab/`.

### Quick local inference test (after first checkpoint)

After 30-60 min of training, the first phase checkpoint will land in Drive. To test it locally:

```powershell
# Wait for drive_sync to mirror it
python OXN\nsos\scripts\drive_sync.py --once

# Then run the chat UI pointing at the new checkpoint
$env:NSOS_CHECKPOINT_DIR = "C:\Users\Oxta\Desktop\reimagined-main\.claude\worktrees\clever-roentgen-ba007c\OXN\nsos\scripts\live_distill_v11_colab\<run_name>"
python OXN\nsos\scripts\chat_v10_ui.py
```

(The v10 chat UI works for any checkpoint with the same architecture format.)

## Troubleshooting

### "No GPU allocated"
Colab free sometimes can't give you a GPU at peak hours. Wait 10 min and retry, or pay $9.99 for 100 compute units.

### "git clone failed: authentication required"
Your repo is private and you didn't set up `GH_TOKEN`. Go back to Pre-flight Step 2.

### "Training OOMs after phase 1"
The 80M profile assumes T4 (16GB). If your session got a V100 (16GB, also fine) or worse, K80 (deprecated), it may OOM. Edit the profile or rerun until you get T4.

### "Checkpoint download is slow on local"
Drive Desktop syncs cloud → local on its own schedule. Check the Drive Desktop tray icon for sync status. Forcing: right-click on the file in Drive Desktop → "Available offline".

### "Bootstrap fails on 'pybind11 not found'"
Bootstrap runs `pip install pybind11>=2.10`. If it fails, in a Colab cell run:
```python
!pip install --upgrade pybind11>=2.10 numpy
```

## What to expect from training

After full curriculum (2-3 sessions):

| Metric | Target |
|---|---|
| Held-out loss (phase 4) | ~3.5-4.5 (vs v10's 7.3) |
| Teacher token accuracy | 55-70% (vs v10's 39%) |
| Coherence of generated text | "Inglês fluente com algum raciocínio" (vs v10's "duo Sub aid 79") |
| Local inference speed | 0.05 tok/s CPU, ~3 tok/s GPU (slightly slower than 40M due to 2× param count) |

## When done

You'll have:
- `Drive:/MyDrive/nsos_v11/runs/v11_t4_80m_run01_2026-05-16/` with all phase checkpoints
- Local mirror in `OXN/nsos/scripts/live_distill_v11_colab/v11_t4_80m_run01_2026-05-16/`
- A trained 80M-param hybrid model that should produce coherent English

Next steps (post-training):
1. **Test inference quality locally** with the chat UI
2. **(Optional) Pacote A.3 + GPU repack** to get faster local inference
3. **(Optional) Distill to 20M** for snappier local chat
