param(
    [string]$RepoRoot = "",
    [string]$Model = "",
    [string]$BuildDir = "",
    [switch]$Cuda
)

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
}
$nsosRoot = Join-Path $RepoRoot "OXN\nsos"
$scriptPath = Join-Path $nsosRoot "scripts\chat_model.py"

if (-not $BuildDir) {
    $candidateMvpBuild = Join-Path $nsosRoot "build-mvp\Release"
    $candidateCudaBuild = Join-Path $nsosRoot "build_cuda129\Release"
    $candidateLegacyBuild = Join-Path $nsosRoot "build_v1\Release"
    if (Test-Path $candidateMvpBuild) {
        $BuildDir = $candidateMvpBuild
    } elseif (Test-Path $candidateCudaBuild) {
        $BuildDir = $candidateCudaBuild
    } else {
        $BuildDir = $candidateLegacyBuild
    }
}

if (-not (Test-Path $scriptPath)) {
    throw "NSOS chat script not found: $scriptPath"
}
if (-not (Test-Path $BuildDir)) {
    throw "NSOS build directory not found: $BuildDir"
}

$arguments = @(
    "-u",
    $scriptPath,
    "--repo-root", $RepoRoot,
    "--build-dir", $BuildDir
)

if ($Model) {
    $arguments += @("--model", $Model)
}

if ($Cuda) {
    $arguments += "--cuda"
}

& python @arguments
exit $LASTEXITCODE
