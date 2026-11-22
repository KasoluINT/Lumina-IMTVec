<#
  一次性轻量化安装（Windows 原生运行，纯 ONNX Runtime DirectML，无需安装 PyTorch / CUDA，也无需额外下载权重）：

      powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
      powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -CpuOnly
#>
param(
    [switch]$CpuOnly,
    [string]$Python = 'python',
    [string]$Venv = '.venv'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "== Lumina-IMTVec (纯 ONNX 轻量化部署，免 PyTorch) ==" -ForegroundColor Cyan
Write-Host "project: $root"
& $Python --version

$venvPython = Join-Path $root "$Venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "creating venv: $Venv"
    & $Python -m venv $Venv
}
& $venvPython -m pip install --upgrade pip --quiet

Write-Host "installing runtime dependencies (onnxruntime-directml, vtracer, shapely, etc.)"
& $venvPython -m pip install -r requirements-native.txt
if ($CpuOnly) {
    & $venvPython -m pip uninstall -y onnxruntime-directml | Out-Null
    & $venvPython -m pip install onnxruntime
}

Write-Host ""
& $venvPython -c "
import onnxruntime as ort
from pathlib import Path
from lumina_vec.vram_guard import compute_runtime_profile
p = compute_runtime_profile()
wdir = Path('weights')
onnx_files = sorted(f.name for f in wdir.glob('*.onnx'))
print('onnxruntime:', ort.__version__, '| providers:', ort.get_available_providers())
print(f'hardware   : GPU={p.gpu_name or \"CPU\"} | VRAM={p.vram_total_mb/1024:.1f}GB (budget {p.vram_budget_mb/1024:.1f}GB) | RAM={p.ram_avail_mb/1024:.1f}/{p.ram_total_mb/1024:.1f}GB')
print(f'vram_guard : sr_tile={p.sr_tile} | evict_per_stage={p.evict_per_stage} | max_work_px={p.max_work_px}')
print(f'weights    : {len(onnx_files)} ONNX models ready -> {onnx_files}')
"
Write-Host "完成。启动：scripts\start_web.bat（或 scripts\start.ps1）" -ForegroundColor Green
