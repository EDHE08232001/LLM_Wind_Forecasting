@echo off
REM =============================================================================
REM Project cleanup for Windows cmd.exe. Safe by default: only caches and OS junk.
REM
REM   cleanup.bat                 caches, .DS_Store / ._* / Thumbs.db, %USERPROFILE%\.matplotlib,
REM                               smoke/demo outputs, synthetic data; frees CUDA memory and lists
REM                               stale project python processes holding the GPU
REM   cleanup.bat --gpu-kill      also terminate those processes (asks first)
REM   cleanup.bat --outputs       also results\predictions, figures, tables (metrics.csv kept)
REM   cleanup.bat --checkpoints   also checkpoints\
REM   cleanup.bat --data          also data\interim and data\processed (raw downloads kept)
REM   cleanup.bat --hf-cache      also the cached Chronos-2 weights
REM   cleanup.bat --pip-cache     also purge pip's cache
REM   cleanup.bat --venv          also delete venvWin
REM   cleanup.bat --all           everything except --venv and raw data
REM =============================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "GPU_KILL=0" & set "OUTPUTS=0" & set "CKPT=0" & set "DATA=0" & set "HF=0" & set "PIPC=0" & set "VENVS=0"
:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--gpu-kill"    set "GPU_KILL=1"
if /i "%~1"=="--outputs"     set "OUTPUTS=1"
if /i "%~1"=="--checkpoints" set "CKPT=1"
if /i "%~1"=="--data"        set "DATA=1"
if /i "%~1"=="--hf-cache"    set "HF=1"
if /i "%~1"=="--pip-cache"   set "PIPC=1"
if /i "%~1"=="--venv"        set "VENVS=1"
if /i "%~1"=="--all" (
    set "OUTPUTS=1" & set "CKPT=1" & set "DATA=1" & set "HF=1" & set "PIPC=1"
)
if /i "%~1"=="--help" ( findstr /b "REM" "%~f0" & exit /b 0 )
shift
goto parse
:parsed

echo ^>^>^> Python / tool caches
for /d /r . %%D in (*) do (
    set "N=%%~nxD"
    if /i "!N!"=="__pycache__" call :rmdir_safe "%%D"
    if /i "!N!"==".pytest_cache" call :rmdir_safe "%%D"
    if /i "!N!"==".ruff_cache" call :rmdir_safe "%%D"
    if /i "!N!"==".mypy_cache" call :rmdir_safe "%%D"
    if /i "!N!"==".ipynb_checkpoints" call :rmdir_safe "%%D"
    if /i "!N:~-9!"==".egg-info" call :rmdir_safe "%%D"
)
for %%D in (build dist htmlcov) do if exist "%%D" ( rmdir /s /q "%%D" & echo   removed %%D )

echo ^>^>^> OS junk (.DS_Store, ._*, Thumbs.db)
for /f "delims=" %%F in ('dir /s /b /a-d ".DS_Store" "._*" "Thumbs.db" "desktop.ini" 2^>nul') do call :del_safe "%%F"

echo ^>^>^> matplotlib caches
for %%D in ("%USERPROFILE%\.matplotlib" "%USERPROFILE%\.cache\matplotlib") do if exist %%D ( rmdir /s /q %%D & echo   removed %%~D )
if defined MPLCONFIGDIR if exist "%MPLCONFIGDIR%" ( rmdir /s /q "%MPLCONFIGDIR%" & echo   removed %MPLCONFIGDIR% )

echo ^>^>^> smoke / demo outputs and synthetic data
for %%D in (results\smoke results\demo data\synthetic checkpoints\smoke chronos-2-finetuned) do if exist "%%D" ( rmdir /s /q "%%D" & echo   removed %%D )

if "%OUTPUTS%"=="1" (
    echo ^>^>^> results outputs ^(metrics.csv kept^)
    for %%D in (results\predictions results\figures results\tables results\runs) do (
        if exist "%%D" (
            for /f "delims=" %%F in ('dir /b /a "%%D" 2^>nul') do if /i not "%%F"==".gitkeep" (
                if exist "%%D\%%F\*" ( rmdir /s /q "%%D\%%F" ) else ( del /f /q "%%D\%%F" )
            )
            echo   cleared %%D
        )
    )
)
if "%CKPT%"=="1" if exist checkpoints ( rmdir /s /q checkpoints & echo   removed checkpoints )
if "%DATA%"=="1" (
    for %%D in (data\interim data\processed) do (
        for /f "delims=" %%F in ('dir /b /a "%%D" 2^>nul') do if /i not "%%F"==".gitkeep" (
            if exist "%%D\%%F\*" ( rmdir /s /q "%%D\%%F" ) else ( del /f /q "%%D\%%F" )
        )
        echo   cleared %%D
    )
)
if "%HF%"=="1" (
    set "HUB=%USERPROFILE%\.cache\huggingface\hub"
    if defined HF_HOME set "HUB=%HF_HOME%\hub"
    if defined HF_HUB_CACHE set "HUB=%HF_HUB_CACHE%"
    if exist "!HUB!\models--amazon--chronos-2" ( rmdir /s /q "!HUB!\models--amazon--chronos-2" & echo   removed Chronos-2 cache )
)

set "PY="
if exist "venvWin\Scripts\python.exe" ( set "PY=venvWin\Scripts\python.exe" ) else ( where python >nul 2>&1 && set "PY=python" )
if "%PIPC%"=="1" if defined PY ( echo ^>^>^> pip cache & "!PY!" -m pip cache purge )

echo ^>^>^> GPU memory (CUDA)
if defined PY (
    if "%GPU_KILL%"=="1" ( "!PY!" scripts\gpu_cleanup.py --kill ) else ( "!PY!" scripts\gpu_cleanup.py )
) else ( echo   ^(no python found, skipped^) )

if "%VENVS%"=="1" if exist venvWin ( rmdir /s /q venvWin & echo   removed venvWin )
echo Cleanup finished.
endlocal
exit /b 0

:rmdir_safe
REM remove a directory unless it lives inside a virtual environment or .git
call :is_protected "%~1" && goto :eof
if exist "%~1" ( rmdir /s /q "%~1" & echo   removed %~1 )
goto :eof

:del_safe
call :is_protected "%~1" && goto :eof
if exist "%~1" ( del /f /q /a "%~1" & echo   removed %~1 )
goto :eof

:is_protected
REM exit code 0 if the path is inside a virtual environment or .git
set "P=%~1"
if not "!P:\venvWin\=!"=="!P!" exit /b 0
if not "!P:\venvMac\=!"=="!P!" exit /b 0
if not "!P:\venvLinux\=!"=="!P!" exit /b 0
if not "!P:\.git\=!"=="!P!" exit /b 0
exit /b 1
