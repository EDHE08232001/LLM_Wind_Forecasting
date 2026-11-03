# Activate venvWin and run the full pipeline; arguments go to scripts\run_all.py.
#   .\run_pipeline.ps1
#   .\run_pipeline.ps1 --config configs/smoke.yaml
#   .\run_pipeline.ps1 --skip finetune
Set-Location -Path $PSScriptRoot
$py = Join-Path $PSScriptRoot "venvWin\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venvWin not found - run .\setup_env.ps1 first" -ForegroundColor Red; exit 1 }
& $py scripts\run_all.py @args
exit $LASTEXITCODE
