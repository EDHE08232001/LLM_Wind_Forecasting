# Run the whole pipeline with the project's virtual environment. Arguments go to scripts\run_all.py.
#   .\run_pipeline.ps1                                  everything (download, zero-shot, LoRA, report)
#   .\run_pipeline.ps1 --config configs/smoke.yaml      quick end-to-end check on synthetic data
#   .\run_pipeline.ps1 --skip download                  the data is already in data\raw
#   .\run_pipeline.ps1 --only report                    rebuild the tables and figures
#   .\run_pipeline.ps1 --method qlora                   QLoRA instead of LoRA (NVIDIA GPU only)
Set-Location -Path $PSScriptRoot
$py = Join-Path $PSScriptRoot "venvWin\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venvWin not found - run .\setup_env.ps1 first" -ForegroundColor Red; exit 1 }
& $py scripts\run_all.py @args
exit $LASTEXITCODE
