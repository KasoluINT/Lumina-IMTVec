@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo   Lumina-IMTVec v1.1.0 - Pure ONNX Vectorization Engine
echo ============================================================

set "PY="
call :try_py "%~dp0.venv\Scripts\python.exe"
if defined PY goto :ready
call :try_py "D:\Lumina-Vec-0.9.2\.venv\Scripts\python.exe"
if defined PY goto :ready
call :try_py "python"
if defined PY goto :ready

echo [First Run] No ready Python ONNX environment found. Running setup...
powershell -ExecutionPolicy Bypass -File "%~dp0scripts\setup_windows.ps1"
if errorlevel 1 (
    echo [Error] Setup failed. Please make sure Python 3.11+ is installed and in PATH.
    pause
    exit /b 1
)
set "PY=%~dp0.venv\Scripts\python.exe"

:ready
set "LVEC_DATA=%CD%\data"
set "PYTHONIOENCODING=utf-8"
if not exist "%LVEC_DATA%" mkdir "%LVEC_DATA%"

echo Python : %PY%
echo Data   : %LVEC_DATA%
echo Starting WebUI at http://127.0.0.1:8091 ...
start "" "http://127.0.0.1:8091"
"%PY%" webui.py --port 8091 %*
if errorlevel 1 pause
exit /b %errorlevel%

:try_py
set "CAND=%~1"
if not "%CAND%"=="python" if not exist "%CAND%" exit /b 0
"%CAND%" -c "import flask, onnxruntime, cv2, skimage, shapely" >nul 2>&1
if not errorlevel 1 set "PY=%CAND%"
exit /b 0
