<#
  build_installer.ps1 — gera o instalador do OContabil de ponta a ponta.

  Passos:
    1. dotnet publish self-contained (win-x64) -> installer\publish
    2. baixa o bootstrapper Evergreen do WebView2 (se ainda não houver) -> installer\redist
    3. compila o instalador com Inno Setup (iscc) -> installer\Output\OContabil-Setup-<versao>.exe

  Requisitos: .NET 8 SDK + Inno Setup 6 (iscc.exe no PATH ou em "C:\Program Files (x86)\Inno Setup 6").
  Uso:  pwsh -File build_installer.ps1
#>
[CmdletBinding()]
param(
  [string]$Configuration = "Release",
  [string]$Rid = "win-x64"
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Definition
$proj = Join-Path $here "..\OContabil\OContabil.csproj" | Resolve-Path
$pub  = Join-Path $here "publish"
$redist = Join-Path $here "redist"

Write-Host "==> Publicando $proj ($Configuration/$Rid, self-contained)..." -ForegroundColor Cyan
dotnet publish $proj -c $Configuration -r $Rid --self-contained true -o $pub --nologo
if ($LASTEXITCODE -ne 0) { throw "dotnet publish falhou ($LASTEXITCODE)" }

New-Item -ItemType Directory -Force $redist | Out-Null
$wv2 = Join-Path $redist "MicrosoftEdgeWebview2Setup.exe"
if (-not (Test-Path $wv2)) {
  Write-Host "==> Baixando bootstrapper do WebView2 Runtime (Evergreen)..." -ForegroundColor Cyan
  try { Invoke-WebRequest -Uri "https://go.microsoft.com/fwlink/p/?LinkId=2124703" -OutFile $wv2 -UseBasicParsing }
  catch { Write-Warning "Não foi possível baixar o WebView2 bootstrapper: $($_.Exception.Message). O instalador seguirá sem ele (o app exibirá instrução de instalação)." }
}

# Localiza o compilador do Inno Setup
$iscc = (Get-Command iscc.exe -ErrorAction SilentlyContinue).Source
if (-not $iscc) {
  foreach ($c in @("C:\Program Files (x86)\Inno Setup 6\ISCC.exe", "C:\Program Files\Inno Setup 6\ISCC.exe")) {
    if (Test-Path $c) { $iscc = $c; break }
  }
}
if (-not $iscc) { throw "Inno Setup (iscc.exe) não encontrado. Instale o Inno Setup 6 e tente novamente." }

Write-Host "==> Compilando instalador com $iscc ..." -ForegroundColor Cyan
& $iscc (Join-Path $here "OContabil.iss")
if ($LASTEXITCODE -ne 0) { throw "iscc falhou ($LASTEXITCODE)" }

Write-Host "==> Instalador gerado em $(Join-Path $here 'Output')" -ForegroundColor Green
