@echo off
rem Copyright 2026 MESFET5198 (Lumina-Vec, Apache License 2.0)
rem Modified 2026 by KasoluINT (Lumina-IMTVec): pure ONNX native WebUI launcher. See LICENSE-APACHE-2.0 & NOTICE.
rem 启动 Lumina-IMTVec Web UI（纯 ONNX 原生运行）。可加参数：scripts\start_web.bat --port 9000
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else if exist "D:\Lumina-Vec-0.9.2\.venv\Scripts\python.exe" (
    set "PY=D:\Lumina-Vec-0.9.2\.venv\Scripts\python.exe"
) else (
    set "PY=python"
)
set "LVEC_DATA=%CD%\data"
set "PYTHONIOENCODING=utf-8"
"%PY%" webui.py %*
if errorlevel 1 pause
