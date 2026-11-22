<#
  启动 Lumina-IMTVec Web UI（纯 ONNX Runtime 原生运行，免 PyTorch）。

      powershell -ExecutionPolicy Bypass -File scripts\start.ps1
      powershell -ExecutionPolicy Bypass -File scripts\start.ps1 -Port 9000
#>
param(
    [int]$Port = 8091,
    [string]$Bind = '0.0.0.0'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$candidates = @(
    (Join-Path $root '.venv\Scripts\python.exe'),
    'D:\Lumina-Vec-0.9.2\.venv\Scripts\python.exe',
    'python'
)
$py = $candidates | Where-Object { $_ -eq 'python' -or (Test-Path $_) } | Select-Object -First 1
if (-not $py) { throw '找不到 Python 环境，请先运行 scripts\setup_windows.ps1' }

$onnxOk = (Test-Path (Join-Path $root 'weights\surface_style.onnx')) -and (Test-Path (Join-Path $root 'weights\feature_part.onnx'))
if (-not $onnxOk) { throw '找不到 weights\*.onnx 模型文件，请确认仓库 weights\ 目录完整' }

$env:LVEC_DATA = Join-Path $root 'data'
$env:PYTHONIOENCODING = 'utf-8'
New-Item -ItemType Directory -Force -Path $env:LVEC_DATA | Out-Null

Write-Host "python : $py"
Write-Host "weights: $(Join-Path $root 'weights')"
Write-Host "data   : $env:LVEC_DATA"
Write-Host "url    : http://127.0.0.1:$Port  (局域网 http://<本机IP>:$Port)"
& $py webui.py --host $Bind --port $Port
