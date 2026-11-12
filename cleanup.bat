@echo off
REM =============================================================================
REM Project cleanup for Windows 11 (cmd.exe). With no options it only removes
REM things that are safe to lose and are rebuilt automatically.
REM
REM   cleanup.bat                 Python and tool caches, .DS_Store / ._* / Thumbs.db files, the
REM                               matplotlib cache, smoke-test outputs and synthetic data. Then
REM                               releases GPU memory and lists project Python processes still running.
REM   cleanup.bat --results       also results\: metrics.csv, tables, figures, predictions
REM   cleanup.bat --checkpoints   also checkpoints\ (the trained LoRA / QLoRA adapters)
REM   cleanup.bat --data          also data\processed (rebuilt by scripts\01_prepare_data.py)
REM   cleanup.bat --raw-data      also data\raw and the kagglehub download cache
REM   cleanup.bat --hf-cache      also the cached Chronos-2 weights (about 0.5 GB, re-downloaded on use)
REM   cleanup.bat --pip-cache     also pip's download cache
REM   cleanup.bat --venv          also venvWin
REM   cleanup.bat --gpu-kill      also stop leftover project Python processes (asks first)
REM   cleanup.bat --all           everything above except --venv and --gpu-kill
REM   cleanup.bat --dry-run       only show what would be removed
REM =============================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "RESULTS=0"
set "CKPT=0"
set "DATA=0"
set "RAW=0"
set "HF=0"
set "PIPC=0"
set "VENVS=0"
set "GPU_KILL=0"
set "DRY=0"

:parse
if "%~1"=="" goto parsed
set "KNOWN=0"
if /i "%~1"=="--results"     ( set "RESULTS=1" & set "KNOWN=1" )
if /i "%~1"=="--checkpoints" ( set "CKPT=1" & set "KNOWN=1" )
if /i "%~1"=="--data"        ( set "DATA=1" & set "KNOWN=1" )
if /i "%~1"=="--raw-data"    ( set "RAW=1" & set "KNOWN=1" )
if /i "%~1"=="--hf-cache"    ( set "HF=1" & set "KNOWN=1" )
if /i "%~1"=="--pip-cache"   ( set "PIPC=1" & set "KNOWN=1" )
if /i "%~1"=="--venv"        ( set "VENVS=1" & set "KNOWN=1" )
if /i "%~1"=="--gpu-kill"    ( set "GPU_KILL=1" & set "KNOWN=1" )
if /i "%~1"=="--dry-run"     ( set "DRY=1" & set "KNOWN=1" )
if /i "%~1"=="--all" (
    set "RESULTS=1" & set "CKPT=1" & set "DATA=1" & set "RAW=1" & set "HF=1" & set "PIPC=1" & set "KNOWN=1"
)
if "!KNOWN!"=="0" (
    if /i not "%~1"=="--help" if /i not "%~1"=="-h" echo Unknown option: %~1
    findstr /b /c:"REM  " "%~f0"
    exit /b 1
)
shift
goto parse
:parsed

echo ^>^>^> Python and tool caches
for /f "delims=" %%D in ('dir /s /b /ad 2^>nul') do (
    set "N=%%~nxD"
    set "HIT=0"
    if /i "!N!"=="__pycache__" set "HIT=1"
    if /i "!N!"==".pytest_cache" set "HIT=1"
    if /i "!N!"==".ruff_cache" set "HIT=1"
    if /i "!N!"==".mypy_cache" set "HIT=1"
    if /i "!N!"==".ipynb_checkpoints" set "HIT=1"
    if /i "!N:~-9!"==".egg-info" set "HIT=1"
    if "!HIT!"=="1" call :remove_unless_protected "%%D"
)
for %%D in (build dist htmlcov .coverage) do call :remove "%%D"

echo ^>^>^> OS junk (.DS_Store, ._*, Thumbs.db, desktop.ini)
for /f "delims=" %%F in ('dir /s /b /a-d ".DS_Store" "._*" "Thumbs.db" "desktop.ini" 2^>nul') do call :remove_unless_protected "%%F"

echo ^>^>^> matplotlib cache
call :remove "%USERPROFILE%\.matplotlib"
call :remove "%USERPROFILE%\.cache\matplotlib"

echo ^>^>^> Smoke-test outputs and synthetic data
for %%D in (results\smoke data\synthetic checkpoints\smoke chronos-2-finetuned) do call :remove "%%D"

if "%RESULTS%"=="1" (
    echo ^>^>^> Results: metrics.csv, tables, figures, predictions
    call :remove "results\metrics.csv"
    call :remove "results\runs"
    for %%D in (results\predictions results\figures results\tables) do call :clear_dir "%%D"
)
if "%CKPT%"=="1" (
    echo ^>^>^> Checkpoints: LoRA / QLoRA adapters
    call :remove "checkpoints"
)
if "%DATA%"=="1" (
    echo ^>^>^> Processed data
    call :clear_dir "data\processed"
)
if "%RAW%"=="1" (
    echo ^>^>^> Raw downloads
    call :clear_dir "data\raw"
    set "KAGGLE=%USERPROFILE%\.cache\kagglehub"
    if defined KAGGLEHUB_CACHE set "KAGGLE=%KAGGLEHUB_CACHE%"
    call :remove "!KAGGLE!\datasets\berkerisen\wind-turbine-scada-dataset"
)
if "%HF%"=="1" (
    echo ^>^>^> Cached Chronos-2 weights
    set "HUB=%USERPROFILE%\.cache\huggingface\hub"
    if defined HF_HOME set "HUB=%HF_HOME%\hub"
    if defined HF_HUB_CACHE set "HUB=%HF_HUB_CACHE%"
    call :remove "!HUB!\models--amazon--chronos-2"
)

set "PY="
if exist "venvWin\Scripts\python.exe" set "PY=venvWin\Scripts\python.exe"
if not defined PY where python >nul 2>&1 && set "PY=python"

if "%PIPC%"=="1" (
    echo ^>^>^> pip cache
    if not defined PY ( echo   no python found, skipped
    ) else if "%DRY%"=="1" ( echo   would run: pip cache purge
    ) else ( "!PY!" -m pip cache purge )
)

echo ^>^>^> GPU memory
if not defined PY (
    echo   no python found, skipped
) else if "%DRY%"=="1" (
    echo   would run: scripts\gpu_cleanup.py
) else if "%GPU_KILL%"=="1" (
    "!PY!" scripts\gpu_cleanup.py --kill
) else (
    "!PY!" scripts\gpu_cleanup.py
)

if "%VENVS%"=="1" (
    echo ^>^>^> Virtual environment
    call :remove "venvWin"
)
echo Cleanup finished.
endlocal
exit /b 0

REM ---------------------------------------------------------------- helpers
:remove
REM delete one file or folder if it exists (or only report it with --dry-run)
if not exist "%~1" goto :eof
if "%DRY%"=="1" (
    echo   would remove %~1
    goto :eof
)
if exist "%~1\" ( rmdir /s /q "%~1" ) else ( del /f /q /a "%~1" )
if exist "%~1" ( echo   could not remove %~1 ) else ( echo   removed %~1 )
goto :eof

:remove_unless_protected
REM same, but never touch anything inside a virtual environment or .git
set "P=%~1\"
if not "!P:\venvWin\=!"=="!P!" goto :eof
if not "!P:\venvMac\=!"=="!P!" goto :eof
if not "!P:\venvLinux\=!"=="!P!" goto :eof
if not "!P:\.git\=!"=="!P!" goto :eof
call :remove "%~1"
goto :eof

:clear_dir
REM empty a tracked folder but keep its .gitkeep
if not exist "%~1" goto :eof
for /f "delims=" %%F in ('dir /b /a "%~1" 2^>nul') do (
    if /i not "%%F"==".gitkeep" call :remove "%~1\%%F"
)
goto :eof
