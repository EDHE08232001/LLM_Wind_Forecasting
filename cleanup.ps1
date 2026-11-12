<#
.SYNOPSIS
  Project cleanup for Windows 11 (PowerShell). With no options it only removes things that are safe
  to lose and are rebuilt automatically.

.DESCRIPTION
  Default: Python and tool caches (__pycache__, .pytest_cache, .ruff_cache, *.egg-info, build\),
  .DS_Store / ._* / Thumbs.db files, the matplotlib cache, smoke-test outputs and synthetic data.
  Then releases GPU memory and lists project Python processes that are still running.

.EXAMPLE
  .\cleanup.ps1                 # defaults
  .\cleanup.ps1 -Results        # also results\: metrics.csv, tables, figures, predictions
  .\cleanup.ps1 -Checkpoints    # also checkpoints\ (the trained LoRA / QLoRA adapters)
  .\cleanup.ps1 -Data           # also data\processed (rebuilt by scripts\01_prepare_data.py)
  .\cleanup.ps1 -RawData        # also data\raw and the kagglehub download cache
  .\cleanup.ps1 -HfCache        # also the cached Chronos-2 weights (~0.5 GB, re-downloaded on use)
  .\cleanup.ps1 -PipCache       # also pip's download cache
  .\cleanup.ps1 -Venv           # also venvWin
  .\cleanup.ps1 -GpuKill        # also stop leftover project Python processes (asks first)
  .\cleanup.ps1 -All            # everything above except -Venv and -GpuKill
  .\cleanup.ps1 -DryRun         # only show what would be removed
#>
param(
    [switch]$Results, [switch]$Checkpoints, [switch]$Data, [switch]$RawData, [switch]$HfCache,
    [switch]$PipCache, [switch]$Venv, [switch]$GpuKill, [switch]$All, [switch]$DryRun
)
Set-Location -Path $PSScriptRoot
if ($All) { $Results = $true; $Checkpoints = $true; $Data = $true; $RawData = $true; $HfCache = $true; $PipCache = $true }

function Remove-Paths([string[]]$paths) {
    foreach ($p in $paths) {
        if (-not $p -or -not (Test-Path -LiteralPath $p)) { continue }
        if ($DryRun) { Write-Host "  would remove $p" }
        else {
            Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction SilentlyContinue
            if (Test-Path -LiteralPath $p) { Write-Host "  could not remove $p (in use?)" -ForegroundColor Yellow }
            else { Write-Host "  removed $p" }
        }
    }
}
function Clear-Dir([string]$dir) {
    # empty a tracked folder but keep its .gitkeep
    if (-not (Test-Path -LiteralPath $dir)) { return }
    Get-ChildItem -LiteralPath $dir -Force | Where-Object { $_.Name -ne ".gitkeep" } |
        ForEach-Object { Remove-Paths @($_.FullName) }
}
$protected = '\\(venvWin|venvMac|venvLinux|\.git)(\\|$)'
$cacheDirs = @("__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".ipynb_checkpoints")
$junkFiles = @(".DS_Store", "Thumbs.db", "desktop.ini")

Write-Host ">>> Python and tool caches"
Get-ChildItem -Recurse -Force -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch $protected -and ($cacheDirs -contains $_.Name -or $_.Name -like "*.egg-info") } |
    ForEach-Object { Remove-Paths @($_.FullName) }
Remove-Paths @("build", "dist", ".coverage", "htmlcov")

Write-Host ">>> OS junk (.DS_Store, ._*, Thumbs.db, desktop.ini)"
Get-ChildItem -Recurse -Force -File -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch $protected -and ($junkFiles -contains $_.Name -or $_.Name -like "._*") } |
    ForEach-Object { Remove-Paths @($_.FullName) }

Write-Host ">>> matplotlib cache"
Remove-Paths @((Join-Path $env:USERPROFILE ".matplotlib"), (Join-Path $env:USERPROFILE ".cache\matplotlib"))

Write-Host ">>> Smoke-test outputs and synthetic data"
Remove-Paths @("results\smoke", "data\synthetic", "checkpoints\smoke", "chronos-2-finetuned")

if ($Results) {
    Write-Host ">>> Results (metrics.csv, tables, figures, predictions)"
    Remove-Paths @("results\metrics.csv", "results\runs")
    foreach ($d in @("results\predictions", "results\figures", "results\tables")) { Clear-Dir $d }
}
if ($Checkpoints) { Write-Host ">>> Checkpoints (LoRA / QLoRA adapters)"; Remove-Paths @("checkpoints") }
if ($Data) { Write-Host ">>> Processed data"; Clear-Dir "data\processed" }
if ($RawData) {
    Write-Host ">>> Raw downloads"
    Clear-Dir "data\raw"
    $kaggle = if ($env:KAGGLEHUB_CACHE) { $env:KAGGLEHUB_CACHE } else { Join-Path $env:USERPROFILE ".cache\kagglehub" }
    Remove-Paths @((Join-Path $kaggle "datasets\berkerisen\wind-turbine-scada-dataset"))
}
if ($HfCache) {
    Write-Host ">>> Cached Chronos-2 weights"
    $hub = if ($env:HF_HUB_CACHE) { $env:HF_HUB_CACHE }
           elseif ($env:HF_HOME) { Join-Path $env:HF_HOME "hub" }
           else { Join-Path $env:USERPROFILE ".cache\huggingface\hub" }
    Remove-Paths @((Join-Path $hub "models--amazon--chronos-2"))
}

$py = $null
if (Test-Path "venvWin\Scripts\python.exe") { $py = "venvWin\Scripts\python.exe" }
else { $cmd = Get-Command python -ErrorAction SilentlyContinue; if ($cmd) { $py = $cmd.Source } }

if ($PipCache) {
    Write-Host ">>> pip cache"
    if (-not $py) { Write-Host "  (no python found, skipped)" }
    elseif ($DryRun) { Write-Host "  would run: $py -m pip cache purge" }
    else { & $py -m pip cache purge }
}

Write-Host ">>> GPU memory"
if (-not $py) { Write-Host "  (no python found, skipped)" }
elseif ($DryRun) { Write-Host "  would run: $py scripts\gpu_cleanup.py" }
elseif ($GpuKill) { & $py scripts\gpu_cleanup.py --kill }
else { & $py scripts\gpu_cleanup.py }

if ($Venv) { Write-Host ">>> Virtual environment"; Remove-Paths @("venvWin") }
Write-Host "Cleanup finished."
