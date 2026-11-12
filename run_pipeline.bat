@echo off
REM Run the whole pipeline with the project's virtual environment. Arguments go to scripts\run_all.py.
REM   run_pipeline.bat                                  everything (download, zero-shot, LoRA, report)
REM   run_pipeline.bat --config configs/smoke.yaml      quick end-to-end check on synthetic data
REM   run_pipeline.bat --skip download                  the data is already in data\raw
REM   run_pipeline.bat --only report                    rebuild the tables and figures
REM   run_pipeline.bat --method qlora                   QLoRA instead of LoRA (NVIDIA GPU only)
cd /d "%~dp0"
if not exist "venvWin\Scripts\python.exe" (
    echo venvWin not found - run setup_env.bat first
    exit /b 1
)
"venvWin\Scripts\python.exe" scripts\run_all.py %*
exit /b %ERRORLEVEL%
