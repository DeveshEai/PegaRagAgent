@echo off
title Pega CLSA Knowledge Assistant
echo ========================================================
echo   Pega CLSA Knowledge Assistant (RAG Web Interface)
echo ========================================================
echo.
set STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
echo Starting Streamlit server...
echo The app will open in your default browser at: http://localhost:8501
echo.
cd /d "%~dp0"
set "VENV_PY=%~dp0..\PEGA-EXP-1\Scripts\python.exe"
if exist "%VENV_PY%" (
    set "PYTHON_EXE=%VENV_PY%"
) else (
    echo [WARN] Virtual environment not found at "%VENV_PY%". Falling back to system python.
    set "PYTHON_EXE=python"
)
"%PYTHON_EXE%" -m streamlit run app.py --server.port=8501 --server.fileWatcherType=none --browser.gatherUsageStats=false
pause
