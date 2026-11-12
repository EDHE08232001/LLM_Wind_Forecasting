<#
.SYNOPSIS
  Environment setup for Windows 11 (PowerShell). Creates .\venvWin with PyTorch (NVIDIA CUDA or CPU),
  Chronos-2, PEFT (LoRA), bitsandbytes (QLoRA, NVIDIA only) and the project itself.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
  .\setup_env.ps1 -Download            # also download the dataset
  .\setup_env.ps1 -Recreate            # delete venvWin and rebuild it
  .\setup_env.ps1 -Cpu                 # force the CPU-only PyTorch wheel
  .\setup_env.ps1 -Cuda cu134          # force a CUDA wheel index
  .\setup_env.ps1 -Python "py -3.13"   # use a specific interpreter

  Afterwards:  .\run_pipeline.ps1      (or activate with .\venvWin\Scripts\Activate.ps1)
#>
param(
    [switch]$Cpu,
    [string]$Cuda = $env:TORCH_CUDA,
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
function Assert-Ok([string]$what) {
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: $what failed." -ForegroundColor Red; exit 1 }
}

# ------------------------------------------------------------------ Python 3.10 - 3.14
if (-not $Python) {
    foreach ($cand in @("py -3.14", "py -3.13", "py -3.12", "py -3.11", "py -3.10", "python", "python3")) {
        if (Test-PyVersion $cand) { $Python = $cand; break }
    }
}
if (-not $Python -or -not (Test-PyVersion $Python)) {
    Write-Host "ERROR: Python 3.10 to 3.14 is required." -ForegroundColor Red
    Write-Host "  Install it with:  winget install Python.Python.3.13   (then open a new terminal)"
    exit 1
}
$pyParts = $Python -split " "
$pyExe = $pyParts[0]; $pyArgs = @($pyParts | Select-Object -Skip 1)
Write-Host ">>> Using $(& $pyExe @pyArgs --version)"

# ------------------------------------------------------------------ virtual environment
if ($Recreate -and (Test-Path $Venv)) { Write-Host ">>> Removing $Venv"; Remove-Item -Recurse -Force $Venv }
$VPy = Join-Path $PSScriptRoot "$Venv\Scripts\python.exe"
if (-not (Test-Path $VPy)) {
    Write-Host ">>> Creating virtual environment .\$Venv"
    & $pyExe @pyArgs -m venv $Venv
    Assert-Ok "Creating the virtual environment"
}
& $VPy -m pip install --upgrade pip setuptools wheel
Assert-Ok "Upgrading pip"

# ------------------------------------------------------------------ PyTorch
function Get-CudaTag {
    # Map the newest CUDA version the NVIDIA driver supports (nvidia-smi header) to a PyTorch wheel index.
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return "" }
    $m = [regex]::Match(((& nvidia-smi) -join "`n"), "CUDA Version:\s*(\d+)\.(\d+)")
    if (-not $m.Success) { return "" }
    $n = [int]$m.Groups[1].Value * 10 + [int]$m.Groups[2].Value
    foreach ($t in @(134, 132, 130, 129, 128, 126, 118)) { if ($n -ge $t) { return "cu$t" } }
    return ""
}

$withQlora = $false
if (-not $Cpu -and -not $Cuda) { $Cuda = Get-CudaTag }
if (-not $Cpu -and $Cuda) {
    Write-Host ">>> NVIDIA GPU -> PyTorch wheel index $Cuda (+ bitsandbytes for QLoRA)"
    & $VPy -m pip install torch --index-url "https://download.pytorch.org/whl/$Cuda"
    $withQlora = $true
} else {
    Write-Host ">>> Installing CPU-only PyTorch (no NVIDIA GPU found, or -Cpu given)"
    & $VPy -m pip install torch --index-url https://download.pytorch.org/whl/cpu
}
Assert-Ok "Installing PyTorch"

# ------------------------------------------------------------------ project
Write-Host ">>> Installing the project (editable)"
& $VPy -m pip install -e ".[dev]"
Assert-Ok "Installing the project"
if ($withQlora) {
    Write-Host ">>> Installing bitsandbytes for QLoRA"
    & $VPy -m pip install -e ".[qlora]"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: bitsandbytes could not be installed - QLoRA will fall back to LoRA." -ForegroundColor Yellow
    }
}

Write-Host ">>> Environment check"
& $VPy scripts\check_env.py
Assert-Ok "The environment check"

if ($Download) {
    Write-Host ">>> Downloading the dataset"
    & $VPy scripts\00_download_data.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: download failed - see README, section 'Dataset', for the manual steps." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Done. Next steps:" -ForegroundColor Green
Write-Host "    .\run_pipeline.ps1                                # the whole experiment"
Write-Host "    .\run_pipeline.ps1 --config configs/smoke.yaml    # quick check on fake data first"
Write-Host "Or activate the environment and run the scripts yourself:"
Write-Host "    .\$Venv\Scripts\Activate.ps1"
Write-Host "    python scripts\run_all.py"
Write-Host "(If activation is blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)"
