@echo off
REM =============================================================================
REM Environment setup for Windows 11 (cmd.exe). Creates venvWin\ with PyTorch
REM (NVIDIA CUDA or CPU), Chronos-2, PEFT (LoRA), bitsandbytes (QLoRA, NVIDIA
REM only) and the project itself.
REM
REM   setup_env.bat                 install everything
REM   setup_env.bat --download      ... and download the dataset
REM   setup_env.bat --recreate      delete venvWin and rebuild it
REM   setup_env.bat --cpu           force the CPU-only PyTorch wheel
REM   setup_env.bat --cuda cu134    force a CUDA wheel index
REM
REM Afterwards:  run_pipeline.bat   (or activate with venvWin\Scripts\activate.bat)
REM =============================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "VENV=venvWin"
set "FORCE_CPU=0"
set "CUDA_TAG=%TORCH_CUDA%"
set "DOWNLOAD=0"
set "RECREATE=0"
set "WITH_QLORA=0"

:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--help" goto usage
if /i "%~1"=="-h" goto usage
if /i "%~1"=="--cpu" ( set "FORCE_CPU=1" & shift & goto parse )
if /i "%~1"=="--download" ( set "DOWNLOAD=1" & shift & goto parse )
if /i "%~1"=="--recreate" ( set "RECREATE=1" & shift & goto parse )
if /i "%~1"=="--cuda" ( set "CUDA_TAG=%~2" & shift & shift & goto parse )
echo Unknown option: %~1
:usage
findstr /b /c:"REM  " "%~f0"
exit /b 1
:parsed

REM ---------------------------------------------------------------- Python 3.10 - 3.14
set "PY="
for %%V in (3.14 3.13 3.12 3.11 3.10) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,15) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo ERROR: Python 3.10 to 3.14 is required.
    echo   Install it with:  winget install Python.Python.3.13   then open a new terminal
    exit /b 1
)
echo ^>^>^> Using %PY%
%PY% --version

REM ---------------------------------------------------------------- virtual environment
if "%RECREATE%"=="1" if exist "%VENV%" (
    echo ^>^>^> Removing %VENV%
    rmdir /s /q "%VENV%"
)
if not exist "%VENV%\Scripts\python.exe" (
    echo ^>^>^> Creating virtual environment %VENV%
    %PY% -m venv "%VENV%" || goto failed
)
set "VPY=%CD%\%VENV%\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip setuptools wheel || goto failed

REM ---------------------------------------------------------------- PyTorch
if "%FORCE_CPU%"=="1" goto torch_cpu
if defined CUDA_TAG goto torch_cuda
where nvidia-smi >nul 2>&1 || goto torch_cpu
REM Read the newest CUDA version the driver supports from the nvidia-smi header ("CUDA Version: 13.0").
set "SMI_LINE="
for /f "tokens=*" %%L in ('nvidia-smi ^| findstr /c:"CUDA Version"') do set "SMI_LINE=%%L"
if not defined SMI_LINE goto torch_cpu
set "SMI_TAIL=!SMI_LINE:*CUDA Version:=!"
set "CUDA_VER="
for /f "tokens=1" %%A in ("!SMI_TAIL!") do set "CUDA_VER=%%A"
set "CUDA_N="
for /f "tokens=1,2 delims=." %%A in ("!CUDA_VER!") do set /a "CUDA_N=%%A*10+%%B"
if not defined CUDA_N goto torch_cpu
for %%T in (134 132 130 129 128 126 118) do (
    if not defined CUDA_TAG if !CUDA_N! GEQ %%T set "CUDA_TAG=cu%%T"
)
if not defined CUDA_TAG goto torch_cpu

:torch_cuda
echo ^>^>^> NVIDIA GPU -^> PyTorch wheel index %CUDA_TAG% (+ bitsandbytes for QLoRA)
"%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/%CUDA_TAG% || goto failed
set "WITH_QLORA=1"
goto project

:torch_cpu
echo ^>^>^> Installing CPU-only PyTorch (no NVIDIA GPU found, or --cpu given)
"%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu || goto failed

REM ---------------------------------------------------------------- project
:project
echo ^>^>^> Installing the project (editable)
"%VPY%" -m pip install -e ".[dev]" || goto failed
if "%WITH_QLORA%"=="1" (
    echo ^>^>^> Installing bitsandbytes for QLoRA
    "%VPY%" -m pip install -e ".[qlora]" || echo WARNING: bitsandbytes could not be installed - QLoRA will fall back to LoRA.
)

echo ^>^>^> Environment check
"%VPY%" scripts\check_env.py || goto failed

if "%DOWNLOAD%"=="1" (
    echo ^>^>^> Downloading the dataset
    "%VPY%" scripts\00_download_data.py || echo WARNING: download failed - see README, section "Dataset", for the manual steps.
)

echo.
echo Done. Next steps:
echo     run_pipeline.bat                                # the whole experiment
echo     run_pipeline.bat --config configs/smoke.yaml    # quick check on fake data first
echo Or activate the environment and run the scripts yourself:
echo     %VENV%\Scripts\activate.bat
echo     python scripts\run_all.py
endlocal
exit /b 0

:failed
echo.
echo ERROR: setup failed - see the messages above.
endlocal
exit /b 1
