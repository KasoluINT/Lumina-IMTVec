@echo off
chcp 65001 >nul
rem Copyright 2026 MESFET5198 (Lumina-Vec, Apache License 2.0)
rem Modified 2026 by KasoluINT (Lumina-IMTVec): pure ONNX native WebUI launcher. See LICENSE-APACHE-2.0 & NOTICE.
setlocal
cd /d "%~dp0.."

set "PY="
call :try_py "%CD%\.venv\Scripts\python.exe"
if defined PY goto :run
call :try_py "D:\Lumina-Vec-0.9.2\.venv\Scripts\python.exe"
if defined PY goto :run
call :try_py "python"
if defined PY goto :run

echo [Error] No ready Python ONNX environment found. Run scripts\setup_windows.ps1 first.
pause
exit /b 1

:run
set "LVEC_DATA=%CD%\data"
set "PYTHONIOENCODING=utf-8"
"%PY%" webui.py %*
if errorlevel 1 pause
exit /b %errorlevel%

:try_py
set "CAND=%~1"
if not "%CAND%"=="python" if not exist "%CAND%" exit /b 0
"%CAND%" -c "import flask, onnxruntime, cv2, skimage, shapely" >nul 2>&1
if not errorlevel 1 set "PY=%CAND%"
exit /b 0
