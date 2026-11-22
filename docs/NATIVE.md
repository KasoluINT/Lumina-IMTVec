# 原生部署（不用 Docker）

把整个项目搬到带 GPU 的 Windows 机器上直接跑。没有容器层，torch 直接拿到显卡，
模型权重和临时文件都落在项目目录里，删掉文件夹就是干净卸载。

小机器上继续用 Docker 也没问题，两种方式互不影响。

## 为什么可以不用 Docker

项目的全部依赖都是普通 Python 包（numpy / OpenCV / vtracer / torch），没有系统级
服务。Docker 在这里唯一的作用是省去装依赖的步骤，代价是容器里的 torch 需要走
WSL2 的虚拟化层，GPU 直通多一层转发。原生跑省掉这一层。

## 前置要求

- Windows 10/11 64 位
- Python 3.10 及以上（`python --version` 能跑）
- NVIDIA 驱动（`nvidia-smi` 能看到卡和 "CUDA Version"）
- 约 4 GB 磁盘（torch + 权重 + 依赖）

## 一步安装

把整个项目文件夹拷到 GPU 机器上，然后在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

它会依次：建 `.venv` 虚拟环境 → 装 CUDA 版 torch → 装其余依赖 → 下载权重（约 390 MB）。

**CUDA 版本要跟驱动匹配。** 先看 `nvidia-smi` 右上角的 "CUDA Version"，然后：

| nvidia-smi 显示 | 参数 |
|---|---|
| CUDA 12.4 及以上 | 默认即可（cu124） |
| CUDA 12.1 | `-Cuda cu121` |
| CUDA 11.8 | `-Cuda cu118` |
| 没有 N 卡 | `-Cuda cpu` |

**Python 3.13 请用 cu126。** cu124 索引里只有 `torch 2.6.0` 支持 cp313，而 cu126 有 2.6～2.14
全套；实测 RTX 4070 SUPER + 驱动 621.01 + `torch 2.8.0+cu126` 正常（`cuda_available=True`）：

```powershell
.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python.exe -m pip install --pre -r requirements-native.txt
.venv\Scripts\python.exe -m pip install "scikit-image<0.27"   # --pre 会把 scikit-image 装成 rc 版
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Cuda cu121
```

## 启动

```bat
scripts\start_web.bat
```

或者直接：

```powershell
.venv\Scripts\python.exe webui.py --port 8091
```

启动时会打印一行 `device : NVIDIA GeForce ...`，看到显卡型号就说明 torch 认到卡了。
如果显示 `CPU (no CUDA GPU visible to torch)`，说明装的 torch 不是 CUDA 版，或者驱动版本不匹配。

## 局域网访问

默认监听 `0.0.0.0:8091`，同一局域网的设备打开 `http://<这台机器IP>:8091` 即可。

第一次访问不通，通常是 Windows 防火墙拦了。放行命令（需要管理员，一次性）：

```powershell
New-NetFirewallRule -DisplayName "Lumina-Vec 8091" -Direction Inbound -Protocol TCP -LocalPort 8091 -Action Allow -Profile Private
```

注意 `-Profile Private`：只对「专用网络」放行。如果网卡被识别成「公用网络」，
要么把网络改成专用，要么去掉这个参数（不推荐）。

## 开机自启（不装服务、不用管理员）

如果希望这台机器作为常驻转换服务，登录后自动起来：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1
```

它只是在你的「启动」文件夹里放一个几 KB 的 `.vbs`，用 `pythonw.exe` 静默运行，
**不注册服务、不写系统启动项、不需要管理员**。取消：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1 -Remove
```

## 数据与权重

| 内容 | 位置 | 覆盖方式 |
|---|---|---|
| 转换记录、上传原图、输出 SVG | `./data` | 环境变量 `LVEC_DATA` |
| 模型权重 | `./weights` | 环境变量 `LVEC_WEIGHTS` |

Docker 版本通过 `compose.yaml` 显式传了 `LVEC_DATA=/data`，两者行为一致。

## 常见问题

**PNG 预览不出来，历史列表没有缩略图。**
Windows 上 `cairosvg` 需要 libcairo（GTK 运行时）。没有的话程序会跳过预览，
**SVG 结果不受影响**，只是列表里少一张缩略图。要恢复：装 GTK3 运行时，再
`pip install cairosvg`。

**权重下载失败。**
GitHub 和 HuggingFace 在国内经常连不上。开代理后设置环境变量再下：

```powershell
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
.venv\Scripts\python.exe scripts\download_weights.py
```

也可以直接从已有 Docker 机器上把 `weights` 文件夹整个拷过来，三个文件共约 375 MB。

**装了 CUDA 版 torch 但还是只认 CPU。**
```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
版本号里带 `+cpu` 就是装错了，重装：
```powershell
.venv\Scripts\python.exe -m pip uninstall -y torch
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**想换端口。**
`scripts\start_web.bat --port 9000`，或设 `LVEC_PORT` 环境变量。
