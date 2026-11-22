<#
  启动 Web UI（原生运行）。

      powershell -ExecutionPolicy Bypass -File scripts\start.ps1
      powershell -ExecutionPolicy Bypass -File scripts\start.ps1 -Port 9000

  Python 环境与权重按顺序找：
    .venv\Scripts\python.exe            （本包的虚拟环境）
    D:\Lumina-Vec-0.9.2\.venv\...       （已有的原项目环境，直接复用）
    权重：.\weights\*.pth  →  D:\Lumina-Vec-0.9.2\weights\*.pth
#>
param(
    [int]$Port = 8091,
    [string]$Bind = '0.0.0.0',
    [string]$Weights = ''
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$candidates = @(
    (Join-Path $root '.venv\Scripts\python.exe'),
    'D:\Lumina-Vec-0.9.2\.venv\Scripts\python.exe'
)
$py = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $py) { throw '找不到 .venv，先运行 scripts\setup_windows.ps1' }

$weightCandidates = @()
if ($Weights) { $weightCandidates += $Weights }
$weightCandidates += (Join-Path $root 'weights\realesr-animevideov3.pth')
$weightCandidates += 'D:\Lumina-Vec-0.9.2\weights\realesr-animevideov3.pth'
$w = $weightCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
$onnxOk = (Test-Path (Join-Path $root 'weights\surface_style.onnx')) -and (Test-Path (Join-Path $root 'weights\feature_part.onnx'))
if (-not $w -and -not $onnxOk) { throw '找不到权重，先运行 scripts\download_weights.py' }

if ($w) { $env:LVEC_WEIGHTS = $w }
$env:LVEC_DATA = Join-Path $root 'data'
$env:PYTHONIOENCODING = 'utf-8'
New-Item -ItemType Directory -Force -Path $env:LVEC_DATA | Out-Null

Write-Host "python : $py"
Write-Host "weights: $w"
Write-Host "data   : $env:LVEC_DATA"
Write-Host "url    : http://127.0.0.1:$Port  (局域网 http://<本机IP>:$Port)"
& $py webui.py --host $Bind --port $Port
