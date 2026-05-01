param(
    [string]$RepoRoot = "",
    [string]$RunName = "small_overnight_v1",
    [string]$Profile = "mamba_small",
    [string]$ResumeModel = "",
    [string]$BuildDir = "",
    [string]$CudaRoot = "",
    [switch]$RebuildCurriculum,
    [switch]$DisableQat,
    [switch]$CudaSync,
    [switch]$PythonFaulthandler,
    [switch]$Cpu,
    [switch]$Background
)

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
}
$nsosRoot = Join-Path $RepoRoot "OXN\nsos"
if (-not $BuildDir) {
    $candidateMvpBuild = Join-Path $nsosRoot "build-mvp\Release"
    $candidateCudaBuild = Join-Path $nsosRoot "build_cuda129\Release"
    $candidateLegacyBuild = Join-Path $nsosRoot "build_v1\Release"
    if (Test-Path $candidateMvpBuild) {
        $buildDir = $candidateMvpBuild
    } elseif (Test-Path $candidateCudaBuild) {
        $buildDir = $candidateCudaBuild
    } else {
        $buildDir = $candidateLegacyBuild
    }
} else {
    $buildDir = $BuildDir
}
$bundleSafe = ($Profile -replace "[^A-Za-z0-9_-]", "_")
$bundleDir = Join-Path $nsosRoot ("artifacts\curriculum_bundle_" + $bundleSafe)
$runDir = Join-Path $nsosRoot ("artifacts\curriculum_runs\" + $RunName)
$logDir = Join-Path $runDir "logs"
$stdoutLog = Join-Path $logDir "stdout.log"
$stderrLog = Join-Path $logDir "stderr.log"
$sessionLog = Join-Path $logDir "session.log"
$pidFile = Join-Path $runDir "train.pid"

New-Item -ItemType Directory -Path $runDir -Force | Out-Null
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$pythonExe = "python"
$scriptPath = Join-Path $nsosRoot "scripts\train_curriculum.py"
$device = if ($Cpu) { "cpu" } else { "gpu" }

if (-not (Test-Path $scriptPath)) {
    throw "NSOS training script not found: $scriptPath"
}
if (-not (Test-Path $buildDir)) {
    throw "NSOS build directory not found: $buildDir"
}
if (-not $CudaRoot -and $env:CUDA_PATH) {
    $CudaRoot = $env:CUDA_PATH
}
if ($CudaRoot -and (Test-Path $CudaRoot)) {
    $env:NSOS_CUDA_ROOT = $CudaRoot
}
if ($CudaSync) {
    $env:NSOS_CUDA_SYNC = "1"
}
if ($PythonFaulthandler) {
    $env:PYTHONFAULTHANDLER = "1"
}

$arguments = @(
    "-u",
    $scriptPath,
    "--profile", $Profile,
    "--bundle-dir", $bundleDir,
    "--run-dir", $runDir,
    "--build-dir", $buildDir,
    "--device", $device,
    "--phase-eval-mode", "fast",
    "--phase-eval-samples", "6",
    "--phase-exact-samples", "2",
    "--phase-best-eval-every-steps", "16",
    "--global-suite-samples-per-phase", "2",
    "--global-suite-exact-samples", "1",
    "--global-suite-every-phases", "2",
    "--replay-ratio", "0.15",
    "--final-consolidation-steps", "0",
    "--checkpoint-every-steps", "16",
    "--progress-mode", "auto",
    "--session-log", $sessionLog
)

if ($RebuildCurriculum) {
    $arguments += "--rebuild-curriculum"
}

if ($DisableQat) {
    $arguments += "--disable-qat"
}

if ($ResumeModel -and (Test-Path $ResumeModel)) {
    $arguments += @("--resume-model", $ResumeModel)
}

if ($Background) {
    $process = Start-Process -FilePath $pythonExe `
        -ArgumentList $arguments `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -WindowStyle Hidden `
        -PassThru

    $process.Id | Set-Content -Path $pidFile -Encoding ascii

    Write-Host "Started NSOS small overnight training in background."
    Write-Host "Run dir : $runDir"
    Write-Host "Profile : $Profile"
    Write-Host "PID     : $($process.Id)"
    Write-Host "Build   : $buildDir"
    if ($env:NSOS_CUDA_ROOT) {
        Write-Host "CUDA    : $env:NSOS_CUDA_ROOT"
    }
    if ($env:NSOS_CUDA_SYNC) {
        Write-Host "Sync    : $env:NSOS_CUDA_SYNC"
    }
    if ($env:PYTHONFAULTHANDLER) {
        Write-Host "FaultH  : $env:PYTHONFAULTHANDLER"
    }
    Write-Host "Stdout  : $stdoutLog"
    Write-Host "Stderr  : $stderrLog"
    Write-Host "Session : $sessionLog"
} else {
    Write-Host "Starting NSOS small overnight training inline."
    Write-Host "Run dir : $runDir"
    Write-Host "Profile : $Profile"
    Write-Host "Build   : $buildDir"
    if ($env:NSOS_CUDA_ROOT) {
        Write-Host "CUDA    : $env:NSOS_CUDA_ROOT"
    }
    if ($env:NSOS_CUDA_SYNC) {
        Write-Host "Sync    : $env:NSOS_CUDA_SYNC"
    }
    if ($env:PYTHONFAULTHANDLER) {
        Write-Host "FaultH  : $env:PYTHONFAULTHANDLER"
    }
    Write-Host "Session : $sessionLog"
    & $pythonExe @arguments
    exit $LASTEXITCODE
}
