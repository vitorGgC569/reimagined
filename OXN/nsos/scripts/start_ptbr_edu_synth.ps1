param(
    [ValidateSet('prepare','run','train')][string]$Action = 'run'
)
$ErrorActionPreference = 'Stop'
$sourceReview = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'data_recipes/ptbr_edu_synth_review.json') -Raw | ConvertFrom-Json
if ($sourceReview.decision -ne 'approved') {
    throw 'Synthetic source failed manual quality review. Training is blocked; inspect data_recipes/ptbr_edu_synth_review.json.'
}
$nsosRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$workspace = Join-Path $nsosRoot 'artifacts/ptbr_edu_synth_20260920_v2'
$python = 'C:/Program Files/WindowsApps/PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0/python3.12.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Python 3.12 unavailable: $python" }
$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -like '*train_ptbr_edu_synth.py*'
}
if ($existing) { throw "A curated training/preparation process is already active: $($existing.ProcessId)" }
New-Item -ItemType Directory -Force -Path $workspace | Out-Null
$env:NSOS_HIP_ROOT = 'C:/TheRock/build'
$env:PATH = 'C:/TheRock/build/bin;' + $env:PATH
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '4'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
$env:NSOS_DETERMINISTIC = '1'
$env:NSOS_TRAIN_CHUNK_SIZE = '1'
$logPrefix = Join-Path $workspace ('job-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$arguments = @('-u', ('"' + (Join-Path $PSScriptRoot 'train_ptbr_edu_synth.py') + '"'), $Action,
    '--build-dir', ('"' + (Join-Path $nsosRoot 'build-gm-hip') + '"'),
    '--workspace', ('"' + $workspace + '"'), '--device', 'gpu',
    '--checkpoint-every-steps', '250', '--checkpoint-every-minutes', '10', '--log-every-steps', '10')
$job = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $nsosRoot `
    -WindowStyle Hidden -RedirectStandardOutput ($logPrefix + '.out.log') `
    -RedirectStandardError ($logPrefix + '.err.log') -PassThru
[pscustomobject]@{Pid=$job.Id; Action=$Action; Workspace=$workspace; Stdout=($logPrefix+'.out.log'); Stderr=($logPrefix+'.err.log')}
