param(
    [ValidateSet('prepare','run','train')][string]$Action = 'run',
    [ValidateSet('baseline','chunked-fp32','chunked-bf16','redesign-fp32','redesign-bf16')][string]$Profile = 'chunked-bf16',
    [ValidateRange(0,1000000)][int]$MaxTrainSteps = 0,
    [string]$RunName = 'main',
    [switch]$DiagnosticTiming,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$nsosRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$workspace = Join-Path $nsosRoot 'artifacts/ptbr_verified_pilot_20260928'
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName must be a simple directory name' }
$python = 'C:/Program Files/WindowsApps/PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0/python3.12.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Python 3.12 unavailable: $python" }
if (-not $DryRun) {
    $existing = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python' -and $_.CommandLine -match 'train_ptbr_(verified_pilot|edu_synth|conversational)\.py'
    }
    if ($existing) { throw "Another PT preparation/training process is active: $($existing.ProcessId)" }
    New-Item -ItemType Directory -Force -Path $workspace | Out-Null
}
$env:NSOS_HIP_ROOT = 'C:/TheRock/build'
$env:PATH = 'C:/TheRock/build/bin;' + $env:PATH
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '4'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
$env:HF_HUB_DOWNLOAD_TIMEOUT = '60'
$env:HF_HUB_ETAG_TIMEOUT = '60'
$env:NSOS_DETERMINISTIC = '1'
$env:NSOS_TRAIN_CHUNK_SIZE = '1'
$env:NSOS_TRAIN_TIMING = $(if ($DiagnosticTiming) { '1' } else { '0' })
$env:NSOS_MAMBA_STAGE_TIMING = '0'
$env:NSOS_MAMBA_FAITHFUL_CHUNKED_FORWARD = $(if ($Profile -eq 'baseline') { '0' } else { '1' })
$env:NSOS_MAMBA_FORWARD_CHUNK_SIZE = '128'
$env:NSOS_MAMBA_BACKWARD_CHUNK_SIZE = $(if ($Profile -eq 'baseline') { '128' } else { '32' })
$precision = if ($Profile.EndsWith('-bf16')) { 'bf16' } else { 'fp32' }
$gpuTrainingProfile = if ($Profile.StartsWith('redesign-')) { 'redesign-v1' } else { 'legacy' }
$logPrefix = Join-Path $workspace ('job-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$arguments = @('-u', ('"' + (Join-Path $PSScriptRoot 'train_ptbr_verified_pilot.py') + '"'), $Action,
    '--build-dir', ('"' + (Join-Path $nsosRoot 'build-gm-hip') + '"'),
    '--workspace', ('"' + $workspace + '"'), '--device', 'gpu', '--matmul-precision', $precision,
    '--gpu-training-profile', $gpuTrainingProfile,
    '--run-dir', ('"' + (Join-Path $workspace ('runs/' + $RunName)) + '"'),
    '--max-train-steps', $MaxTrainSteps, '--checkpoint-every-steps', '500',
    '--checkpoint-every-minutes', '10', '--log-every-steps', '20')
if ($DryRun) {
    [pscustomobject]@{Action=$Action; Profile=$Profile; Precision=$precision;
        GpuTrainingProfile=$gpuTrainingProfile; DiagnosticTiming=[bool]$DiagnosticTiming;
        Run=$RunName; Python=$python; Arguments=$arguments; WorkingDirectory=$nsosRoot;
        TrainingTiming=$env:NSOS_TRAIN_TIMING; LayerTiming=$env:NSOS_MAMBA_STAGE_TIMING;
        ForwardChunked=$env:NSOS_MAMBA_FAITHFUL_CHUNKED_FORWARD;
        ForwardChunkSize=$env:NSOS_MAMBA_FORWARD_CHUNK_SIZE;
        BackwardChunkSize=$env:NSOS_MAMBA_BACKWARD_CHUNK_SIZE} | ConvertTo-Json -Depth 4
    return
}
$job = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $nsosRoot `
    -WindowStyle Hidden -RedirectStandardOutput ($logPrefix + '.out.log') `
    -RedirectStandardError ($logPrefix + '.err.log') -PassThru
[pscustomobject]@{Pid=$job.Id; Action=$Action; Profile=$Profile; Workspace=$workspace; Run=$RunName; Stdout=($logPrefix+'.out.log'); Stderr=($logPrefix+'.err.log')}
