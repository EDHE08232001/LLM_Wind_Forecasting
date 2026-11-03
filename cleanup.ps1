<#
.SYNOPSIS
  Project cleanup for Windows (PowerShell). Safe by default: only caches and OS junk.

.DESCRIPTION
  Default: __pycache__, .pytest_cache, .ruff_cache, *.egg-info, build/, .DS_Store / ._* / Thumbs.db,
  the matplotlib cache (%USERPROFILE%\.matplotlib), smoke/demo outputs and synthetic data, then frees
  CUDA memory and lists stale project python processes holding the GPU.

.EXAMPLE
  .\cleanup.ps1                 # defaults
  .\cleanup.ps1 -GpuKill        # also terminate stale project python processes (asks first)
  .\cleanup.ps1 -Outputs        # also results\predictions, figures, tables (metrics.csv kept)
  .\cleanup.ps1 -Checkpoints    # also checkpoints\
  .\cleanup.ps1 -Data           # also data\interim and data\processed (raw downloads kept)
  .\cleanup.ps1 -HfCache        # also the cached Chronos-2 weights
  .\cleanup.ps1 -PipCache       # also purge pip's cache
  .\cleanup.ps1 -Venv           # also delete venvWin
  .\cleanup.ps1 -All            # everything except -Venv and raw data
  .\cleanup.ps1 -DryRun         # show what would be removed
#>
param(
    [switch]$GpuKill, [switch]$Outputs, [switch]$Checkpoints, [switch]$Data,
    [switch]$HfCache, [switch]$PipCache, [switch]$Venv, [switch]$All, [switch]$DryRun
)
Set-Location -Path $PSScriptRoot
if ($All) { $Outputs = $true; $Checkpoints = $true; $Data = $true; $HfCache = $true; $PipCache = $true }

function Remove-Paths([string[]]$paths) {
    foreach ($p in $paths) {
        if (-not $p) { continue }
        if (Test-Path -LiteralPath $p) {
            if ($DryRun) { Write-Host "  would remove $p" }
            else {
                Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction SilentlyContinue
                Write-Host "  removed $p"
            }
        }
    }
}
function Clear-KeepGitkeep([string]$dir) {
    if (-not (Test-Path $dir)) { return }
    Get-ChildItem -LiteralPath $dir -Force | Where-Object { $_.Name -ne ".gitkeep" } |
        ForEach-Object { Remove-Paths @($_.FullName) }
}
$skip = '\\(venvWin|venvMac|venvLinux|\.git)(\\|$)'

Write-Host ">>> Python / tool caches"
Get-ChildItem -Recurse -Force -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch $skip -and $_.Name -in @("__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".ipynb_checkpoints") -or ($_.Name -like "*.egg-info" -and $_.FullName -notmatch $skip) } |
    ForEach-Object { Remove-Paths @($_.FullName) }
Remove-Paths @("build", "dist", ".coverage", "htmlcov")

Write-Host ">>> OS junk (.DS_Store, ._*, Thumbs.db, desktop.ini)"
Get-ChildItem -Recurse -Force -File -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch $skip -and ($_.Name -eq ".DS_Store" -or $_.Name -like "._*" -or $_.Name -eq "Thumbs.db" -or $_.Name -eq "desktop.ini") } |
    ForEach-Object { Remove-Paths @($_.FullName) }

Write-Host ">>> matplotlib caches"
Remove-Paths @((Join-Path $env:USERPROFILE ".matplotlib"), (Join-Path $env:USERPROFILE ".cache\matplotlib"))
if ($env:MPLCONFIGDIR) { Remove-Paths @($env:MPLCONFIGDIR) }

Write-Host ">>> smoke / demo outputs and synthetic data"
Remove-Paths @("results\smoke", "results\demo", "data\synthetic", "checkpoints\smoke", "chronos-2-finetuned")

if ($Outputs) {
    Write-Host ">>> results outputs (metrics.csv kept)"
    foreach ($d in @("results\predictions", "results\figures", "results\tables", "results\runs")) { Clear-KeepGitkeep $d }
}
if ($Checkpoints) { Write-Host ">>> checkpoints"; Remove-Paths @("checkpoints") }
if ($Data) { Write-Host ">>> derived data (raw kept)"; Clear-KeepGitkeep "data\interim"; Clear-KeepGitkeep "data\processed" }
if ($HfCache) {
    Write-Host ">>> Hugging Face cache for Chronos-2"
    $hub = if ($env:HF_HUB_CACHE) { $env:HF_HUB_CACHE } elseif ($env:HF_HOME) { Join-Path $env:HF_HOME "hub" } else { Join-Path $env:USERPROFILE ".cache\huggingface\hub" }
    Remove-Paths @((Join-Path $hub "models--amazon--chronos-2"))
}

$py = if (Test-Path "venvWin\Scripts\python.exe") { "venvWin\Scripts\python.exe" } else { (Get-Command python -ErrorAction SilentlyContinue).Source }
if ($PipCache -and $py) {
    Write-Host ">>> pip cache"
    if ($DryRun) { Write-Host "  would run: pip cache purge" } else { & $py -m pip cache purge }
}

Write-Host ">>> GPU memory (CUDA)"
if ($py -and -not $DryRun) {
    if ($GpuKill) { & $py scripts\gpu_cleanup.py --kill } else { & $py scripts\gpu_cleanup.py }
} else { Write-Host "  (skipped)" }

if ($Venv) { Write-Host ">>> virtual environment"; Remove-Paths @("venvWin") }
Write-Host "Cleanup finished."
