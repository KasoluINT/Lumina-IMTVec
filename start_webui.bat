@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo   Lumina-IMTVec v1.1.0 - Pure ONNX Vectorization Engine
echo ============================================================

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    echo [First Run] Local .venv not found. Running pure-ONNX setup...
    powershell -ExecutionPolicy Bypass -File "%~dp0scripts\setup_windows.ps1"
    if errorlevel 1 (
        echo [Error] Setup failed. Please make sure Python 3.11+ is installed and in PATH.
        pause
        exit /b 1
    )
    set "PY=.venv\Scripts\python.exe"
)

set "LVEC_DATA=%CD%\data"
set "PYTHONIOENCODING=utf-8"
if not exist "%LVEC_DATA%" mkdir "%LVEC_DATA%"

echo Starting WebUI at http://127.0.0.1:8091 ...
start "" "http://127.0.0.1:8091"
"%PY%" webui.py --port 8091 %*
if errorlevel 1 pause
