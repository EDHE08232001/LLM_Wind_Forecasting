@echo off
REM =============================================================================
REM Environment setup for Windows cmd.exe (NVIDIA CUDA or CPU). Creates venvWin\
REM
REM   setup_env.bat                 auto-detect GPU, install everything
REM   setup_env.bat --download      ... and download the dataset
REM   setup_env.bat --cpu           force CPU-only PyTorch
REM   setup_env.bat --cuda cu134    force a CUDA wheel index
REM   setup_env.bat --no-chronos    skip Chronos-2 / transformers / peft
REM   setup_env.bat --qlora         also install bitsandbytes (4-bit QLoRA)
REM   setup_env.bat --recreate      delete and rebuild venvWin
REM
REM Afterwards:  venvWin\Scripts\activate.bat
REM =============================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "VENV=venvWin"
set "FORCE_CPU=0"
set "CUDA_TAG=%TORCH_CUDA%"
set "WITH_CHRONOS=1"
set "WITH_QLORA=0"
set "DOWNLOAD=0"
set "RECREATE=0"

:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--cpu"        set "FORCE_CPU=1"
if /i "%~1"=="--no-chronos" set "WITH_CHRONOS=0"
if /i "%~1"=="--qlora"      set "WITH_QLORA=1"
if /i "%~1"=="--download"   set "DOWNLOAD=1"
if /i "%~1"=="--recreate"   set "RECREATE=1"
if /i "%~1"=="--cuda" (
    set "CUDA_TAG=%~2"
    shift
)
if /i "%~1"=="--help" goto usage
if /i "%~1"=="-h" goto usage
shift
goto parse
:usage
findstr /b "REM" "%~f0"
exit /b 0
:parsed

REM ---------------------------------------------------------------- python
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
    echo ERROR: need Python 3.10-3.14 ^(3.14 recommended^).
    echo   Install from https://www.python.org/downloads/  or:  winget install Python.Python.3.14
    exit /b 1
)
echo ^>^>^> Using %PY%
%PY% --version

REM ---------------------------------------------------------------- venv
if "%RECREATE%"=="1" if exist "%VENV%" (
    echo ^>^>^> Removing %VENV%
    rmdir /s /q "%VENV%"
)
if not exist "%VENV%\Scripts\python.exe" (
    echo ^>^>^> Creating virtual environment %VENV%
    %PY% -m venv "%VENV%" || exit /b 1
)
set "VPY=%CD%\%VENV%\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip setuptools wheel || exit /b 1

REM ---------------------------------------------------------------- torch
if "%FORCE_CPU%"=="1" goto torch_cpu
if defined CUDA_TAG goto torch_cuda
where nvidia-smi >nul 2>&1 || goto torch_cpu
set "CUDA_VER="
set "SMI_LINE="
for /f "tokens=*" %%L in ('nvidia-smi ^| findstr /c:"CUDA Version"') do set "SMI_LINE=%%L"
if not defined SMI_LINE goto torch_cpu
REM keep only the text after "CUDA Version:" (e.g. " 13.0   |") and take the first token
set "SMI_TAIL=!SMI_LINE:*CUDA Version:=!"
for /f "tokens=1" %%A in ("!SMI_TAIL!") do set "CUDA_VER=%%A"
for /f "tokens=1,2 delims=." %%A in ("!CUDA_VER!") do (
    set /a "CUDA_N=%%A*10+%%B"
)
if not defined CUDA_N goto torch_cpu
if !CUDA_N! GEQ 134 ( set "CUDA_TAG=cu134" ) else if !CUDA_N! GEQ 132 ( set "CUDA_TAG=cu132" ) else if !CUDA_N! GEQ 130 ( set "CUDA_TAG=cu130" ) else if !CUDA_N! GEQ 129 ( set "CUDA_TAG=cu129" ) else if !CUDA_N! GEQ 128 ( set "CUDA_TAG=cu128" ) else if !CUDA_N! GEQ 126 ( set "CUDA_TAG=cu126" ) else if !CUDA_N! GEQ 118 ( set "CUDA_TAG=cu118" )
if not defined CUDA_TAG goto torch_cpu

:torch_cuda
echo ^>^>^> NVIDIA GPU -^> PyTorch wheel index %CUDA_TAG%
"%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/%CUDA_TAG% || exit /b 1
goto project

:torch_cpu
echo ^>^>^> Installing CPU-only PyTorch
"%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu || exit /b 1

REM ---------------------------------------------------------------- project
:project
set "EXTRAS=dev"
if "%WITH_CHRONOS%"=="1" set "EXTRAS=chronos,%EXTRAS%"
if "%WITH_QLORA%"=="1" set "EXTRAS=%EXTRAS%,qlora"
echo ^>^>^> Installing project (editable) with extras [%EXTRAS%]
"%VPY%" -m pip install -e ".[%EXTRAS%]" || exit /b 1

echo ^>^>^> Environment check
"%VPY%" scripts\check_env.py

if "%DOWNLOAD%"=="1" (
    echo ^>^>^> Downloading dataset
    "%VPY%" scripts\00_download_data.py || echo !!! Download failed - see README "Dataset" for manual steps.
)

echo.
echo Done. Activate the environment with:
echo     %VENV%\Scripts\activate.bat        (cmd.exe)
echo     .\%VENV%\Scripts\Activate.ps1      (PowerShell)
echo Then e.g.:  python scripts\run_all.py   ^|   python scripts\demo.py --synthetic
endlocal
