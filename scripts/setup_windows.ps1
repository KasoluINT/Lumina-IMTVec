<#
  一次性安装（Windows，原生运行，不用 Docker）。

      powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
      powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Cuda cu128
      powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Cuda cpu

  -Cuda 选 torch 构建：cu126（默认，Python 3.13 只有它覆盖 2.6~2.14）、
  cu124（只到 2.6.0）、cu128、cpu。先看 nvidia-smi 右上角的 CUDA Version。
#>
param(
    [ValidateSet('cu126', 'cu124', 'cu128', 'cpu')] [string]$Cuda = 'cu126',
    [string]$Python = 'python',
    [string]$Venv = '.venv'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "== Lumina-Vec (Lumina 精确描摹版) ==" -ForegroundColor Cyan
Write-Host "project: $root"
& $Python --version

$venvPython = Join-Path $root "$Venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "creating venv: $Venv"
    & $Python -m venv $Venv
}
& $venvPython -m pip install --upgrade pip --quiet

if ($Cuda -eq 'cpu') {
    & $venvPython -m pip install torch --index-url https://download.pytorch.org/whl/cpu
} else {
    & $venvPython -m pip install torch --index-url "https://download.pytorch.org/whl/$Cuda"
}
if ($LASTEXITCODE -ne 0) { throw "torch 安装失败" }

Write-Host "installing runtime dependencies"
& $venvPython -m pip install --pre -r requirements-native.txt
& $venvPython -m pip install "scikit-image<0.27"   # --pre 会把 scikit-image 装成 rc 版

Write-Host "downloading model weights (~390 MB)"
& $venvPython scripts\download_weights.py

Write-Host ""
& $venvPython -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
Write-Host "完成。启动：scripts\\start_web.bat（或 scripts\\start.ps1）" -ForegroundColor Green
