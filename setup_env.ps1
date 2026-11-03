<#
.SYNOPSIS
  Environment setup for Windows (NVIDIA CUDA or CPU). Creates .\venvWin

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
  .\setup_env.ps1 -Download            # also download the dataset
  .\setup_env.ps1 -Cpu                 # force CPU-only PyTorch
  .\setup_env.ps1 -Cuda cu134          # force a CUDA wheel index
  .\setup_env.ps1 -NoChronos           # skip Chronos-2 / transformers / peft
  .\setup_env.ps1 -QLoRA               # also install bitsandbytes (4-bit QLoRA)
  .\setup_env.ps1 -Python "py -3.14" -Recreate

  Afterwards:  .\venvWin\Scripts\Activate.ps1
  (If activation is blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)
#>
param(
    [switch]$Cpu,
    [string]$Cuda = $env:TORCH_CUDA,
    [switch]$NoChronos,
    [switch]$QLoRA,
    [switch]$Download,
    [switch]$Recreate,
    [string]$Python = ""
)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$Venv = "venvWin"

function Test-PyVersion([string]$cmd) {
    try {
        $parts = $cmd -split " "
        $exe = $parts[0]; $rest = @($parts | Select-Object -Skip 1)
        & $exe @rest -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,15) else 1)" 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

# ------------------------------------------------------------------ python
if (-not $Python) {
    foreach ($cand in @("py -3.14", "py -3.13", "py -3.12", "py -3.11", "py -3.10", "python", "python3")) {
        if (Test-PyVersion $cand) { $Python = $cand; break }
    }
}
if (-not $Python -or -not (Test-PyVersion $Python)) {
    Write-Host "ERROR: need Python 3.10-3.14 (3.14 recommended). Install from https://www.python.org/downloads/ or: winget install Python.Python.3.14" -ForegroundColor Red
    exit 1
}
$pyParts = $Python -split " "
$pyExe = $pyParts[0]; $pyArgs = @($pyParts | Select-Object -Skip 1)
Write-Host ">>> Using $(& $pyExe @pyArgs --version)"

# ------------------------------------------------------------------ venv
if ($Recreate -and (Test-Path $Venv)) { Write-Host ">>> Removing $Venv"; Remove-Item -Recurse -Force $Venv }
if (-not (Test-Path $Venv)) {
    Write-Host ">>> Creating virtual environment .\$Venv"
    & $pyExe @pyArgs -m venv $Venv
}
$VPy = Join-Path $PSScriptRoot "$Venv\Scripts\python.exe"
& $VPy -m pip install --upgrade pip setuptools wheel

# ------------------------------------------------------------------ torch
function Get-CudaTag {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) { return "" }
    $out = (& nvidia-smi) -join "`n"
    $m = [regex]::Match($out, "CUDA Version:\s*(\d+)\.(\d+)")
    if (-not $m.Success) { return "" }
    $n = [int]$m.Groups[1].Value * 10 + [int]$m.Groups[2].Value
    if ($n -ge 134) { return "cu134" }
    if ($n -ge 132) { return "cu132" }
    if ($n -ge 130) { return "cu130" }
    if ($n -ge 129) { return "cu129" }
    if ($n -ge 128) { return "cu128" }
    if ($n -ge 126) { return "cu126" }
    if ($n -ge 118) { return "cu118" }
    return ""
}

if ($Cpu) {
    Write-Host ">>> Installing CPU-only PyTorch"
    & $VPy -m pip install torch --index-url https://download.pytorch.org/whl/cpu
} else {
    if (-not $Cuda) { $Cuda = Get-CudaTag }
    if ($Cuda) {
        Write-Host ">>> NVIDIA GPU detected -> PyTorch wheel index $Cuda"
        & $VPy -m pip install torch --index-url "https://download.pytorch.org/whl/$Cuda"
    } else {
        Write-Host ">>> No NVIDIA GPU detected -> CPU-only PyTorch"
        & $VPy -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    }
}
if ($LASTEXITCODE -ne 0) { Write-Host "PyTorch install failed" -ForegroundColor Red; exit 1 }

# ------------------------------------------------------------------ project
$extras = "dev"
if (-not $NoChronos) { $extras = "chronos,$extras" }
if ($QLoRA) { $extras = "$extras,qlora" }
Write-Host ">>> Installing project (editable) with extras [$extras]"
& $VPy -m pip install -e ".[$extras]"
if ($LASTEXITCODE -ne 0) { Write-Host "Project install failed" -ForegroundColor Red; exit 1 }

Write-Host ">>> Environment check"
& $VPy scripts\check_env.py

if ($Download) {
    Write-Host ">>> Downloading dataset"
    & $VPy scripts\00_download_data.py
    if ($LASTEXITCODE -ne 0) { Write-Host "!!! Download failed - see README 'Dataset' for manual steps." -ForegroundColor Yellow }
}

Write-Host ""
Write-Host "Done. Activate the environment with:" -ForegroundColor Green
Write-Host "    .\$Venv\Scripts\Activate.ps1        (PowerShell)"
Write-Host "    $Venv\Scripts\activate.bat          (cmd.exe)"
Write-Host "Then e.g.:  python scripts\run_all.py   |   python scripts\demo.py --synthetic"
