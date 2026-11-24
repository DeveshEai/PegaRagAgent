@echo off
title Pega Academy Complete 200+ Downloader
echo ========================================================
echo   Pega Academy - Complete 200+ Topics & Challenges
echo ========================================================
echo.
cd /d "%~dp0"
set "VENV_PY=%~dp0..\PEGA-EXP-1\Scripts\python.exe"
if exist "%VENV_PY%" (
    set "PYTHON_EXE=%VENV_PY%"
) else (
    echo [WARN] Virtual environment not found at "%VENV_PY%". Falling back to system python.
    set "PYTHON_EXE=python"
)
echo [1/2] Running curriculum module downloader (pega_downloader.py)...
"%PYTHON_EXE%" pega_downloader.py
if errorlevel 1 (
    echo [ERROR] pega_downloader.py exited with error code %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)
echo.
echo [2/2] Running hands-on challenges downloader (download_challenges.py)...
"%PYTHON_EXE%" download_challenges.py
if errorlevel 1 (
    echo [ERROR] download_challenges.py exited with error code %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)
echo.
pause
