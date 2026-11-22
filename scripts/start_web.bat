@echo off
rem 启动 Web UI（原生运行）。可加参数：scripts\start_web.bat --port 9000
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else if exist "D:\Lumina-Vec-0.9.2\.venv\Scripts\python.exe" (
    set "PY=D:\Lumina-Vec-0.9.2\.venv\Scripts\python.exe"
) else (
    set "PY=python"
)
if exist "weights\realesr-animevideov3.pth" set "LVEC_WEIGHTS=%CD%\weights\realesr-animevideov3.pth"
if not exist "%LVEC_WEIGHTS%" if exist "D:\Lumina-Vec-0.9.2\weights\realesr-animevideov3.pth" set "LVEC_WEIGHTS=D:\Lumina-Vec-0.9.2\weights\realesr-animevideov3.pth"
set "LVEC_DATA=%CD%\data"
set "PYTHONIOENCODING=utf-8"
"%PY%" webui.py %*
if errorlevel 1 pause
