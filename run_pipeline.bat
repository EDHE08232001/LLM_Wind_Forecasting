@echo off
REM Activate venvWin and run the full pipeline; arguments go to scripts\run_all.py.
REM   run_pipeline.bat
REM   run_pipeline.bat --config configs/smoke.yaml
cd /d "%~dp0"
if not exist "venvWin\Scripts\python.exe" (
    echo venvWin not found - run setup_env.bat first
    exit /b 1
)
"venvWin\Scripts\python.exe" scripts\run_all.py %*
