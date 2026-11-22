"""Zero-PyTorch dynamic GPU VRAM & system RAM probing for safe low-memory ONNX execution."""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
import ctypes.wintypes as wt
import os
import subprocess


@dataclass(frozen=True)
class RuntimeProfile:
    gpu_name: str | None
    vram_total_mb: float
    vram_budget_mb: float
    ram_total_mb: float
    ram_avail_mb: float
    sr_tile: int
    sr_pad: int
    surface_batch: int
    evict_per_stage: bool
    force_cpu: bool
    max_work_px: int
    max_zoom_px: int


def query_system_ram_mb() -> tuple[float, float]:
    """Return `(total_ram_mb, avail_ram_mb)` via Windows kernel32 or safe default."""
    if os.name == "nt":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wt.DWORD),
                    ("dwMemoryLoad", wt.DWORD),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            ms = MEMORYSTATUSEX()
            ms.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                mb = float(1024 * 1024)
                return ms.ullTotalPhys / mb, ms.ullAvailPhys / mb
        except Exception:
            pass
    return 16384.0, 8192.0


def _query_dxgi_vram_mb() -> tuple[str | None, float, float]:
    """Enumerate Windows DXGI hardware adapters (NVIDIA/AMD/Intel) with zero PyTorch."""
    if os.name != "nt":
        return None, 0.0, 0.0
    try:
        class GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", wt.DWORD),
                ("Data2", wt.WORD),
                ("Data3", wt.WORD),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        class LUID(ctypes.Structure):
            _fields_ = [("LowPart", wt.DWORD), ("HighPart", wt.LONG)]

        class DXGI_ADAPTER_DESC1(ctypes.Structure):
            _fields_ = [
                ("Description", wt.WCHAR * 128),
                ("VendorId", wt.UINT),
                ("DeviceId", wt.UINT),
                ("SubSysId", wt.UINT),
                ("Revision", wt.UINT),
                ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", LUID),
                ("Flags", wt.UINT),
            ]

        class DXGI_QUERY_VIDEO_MEMORY_INFO(ctypes.Structure):
            _fields_ = [
                ("Budget", ctypes.c_uint64),
                ("CurrentUsage", ctypes.c_uint64),
                ("AvailableForReservation", ctypes.c_uint64),
                ("CurrentReservation", ctypes.c_uint64),
            ]

        iid_factory1 = GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
        iid_adapter3 = GUID(0x645967A4, 0x1392, 0x4310, (ctypes.c_ubyte * 8)(0xA7, 0x98, 0x80, 0x53, 0xCE, 0x3E, 0x93, 0xFD))
        dxgi = ctypes.windll.dxgi
        factory = ctypes.c_void_p()
        if dxgi.CreateDXGIFactory1(ctypes.byref(iid_factory1), ctypes.byref(factory)) != 0 or not factory.value:
            return None, 0.0, 0.0

        vtbl = ctypes.cast(ctypes.cast(factory, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
        enum_fn = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wt.UINT, ctypes.POINTER(ctypes.c_void_p))(vtbl[12])
        release_factory = ctypes.WINFUNCTYPE(wt.ULONG, ctypes.c_void_p)(vtbl[2])

        best_name: str | None = None
        best_total_mb = 0.0
        best_avail_mb = 0.0
        mb = float(1024 * 1024)
        idx = 0
        try:
            while True:
                adp = ctypes.c_void_p()
                if enum_fn(factory, idx, ctypes.byref(adp)) != 0 or not adp.value:
                    break
                adp_vtbl = ctypes.cast(ctypes.cast(adp, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
                desc_fn = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(DXGI_ADAPTER_DESC1))(adp_vtbl[10])
                qi_fn = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))(adp_vtbl[0])
                release_adp = ctypes.WINFUNCTYPE(wt.ULONG, ctypes.c_void_p)(adp_vtbl[2])

                desc = DXGI_ADAPTER_DESC1()
                if desc_fn(adp, ctypes.byref(desc)) == 0 and (desc.Flags & 2) == 0:
                    ded_mb = float(desc.DedicatedVideoMemory) / mb
                    avail_mb = max(0.0, ded_mb * 0.72)
                    # Try IDXGIAdapter3::QueryVideoMemoryInfo (slot 14) for real live free VRAM budget
                    adp3 = ctypes.c_void_p()
                    if qi_fn(adp, ctypes.byref(iid_adapter3), ctypes.byref(adp3)) == 0 and adp3.value:
                        try:
                            a3_vtbl = ctypes.cast(ctypes.cast(adp3, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
                            qvm_fn = ctypes.WINFUNCTYPE(
                                ctypes.c_long, ctypes.c_void_p, wt.UINT, wt.UINT, ctypes.POINTER(DXGI_QUERY_VIDEO_MEMORY_INFO)
                            )(a3_vtbl[14])
                            mem_info = DXGI_QUERY_VIDEO_MEMORY_INFO()
                            if qvm_fn(adp3, 0, 0, ctypes.byref(mem_info)) == 0 and mem_info.Budget > 0:
                                live_free = max(0.0, float(mem_info.Budget) - float(mem_info.CurrentUsage)) / mb
                                avail_mb = min(ded_mb * 0.85, max(live_free * 0.85, ded_mb * 0.45))
                            ctypes.WINFUNCTYPE(wt.ULONG, ctypes.c_void_p)(a3_vtbl[2])(adp3)
                        except Exception:
                            pass
                    if ded_mb > best_total_mb:
                        best_name = str(desc.Description).strip()
                        best_total_mb = ded_mb
                        best_avail_mb = avail_mb
                release_adp(adp)
                idx += 1
        finally:
            release_factory(factory)
        return best_name, best_total_mb, best_avail_mb
    except Exception:
        return None, 0.0, 0.0


def query_gpu_vram_mb() -> tuple[str | None, float, float]:
    """Return `(gpu_name, total_vram_mb, safe_avail_vram_mb)` without importing PyTorch."""
    name, total_mb, avail_mb = _query_dxgi_vram_mb()
    if name is None or total_mb <= 0.0:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                text=True,
                timeout=3,
                stderr=subprocess.DEVNULL,
            ).strip()
            if out:
                first = out.splitlines()[0].split(",")
                if len(first) >= 3:
                    name = first[0].strip()
                    total_mb = float(first[1].strip())
                    avail_mb = float(first[2].strip()) * 0.85
        except Exception:
            pass

    lim_raw = os.environ.get("LUMINA_VRAM_LIMIT_GB", "").strip()
    if lim_raw:
        try:
            lim_mb = max(256.0, float(lim_raw) * 1024.0)
            avail_mb = min(avail_mb if avail_mb > 0 else lim_mb, lim_mb)
            if total_mb <= 0:
                total_mb = lim_mb
        except ValueError:
            pass
    return name, round(total_mb, 1), round(avail_mb, 1)


def compute_runtime_profile() -> RuntimeProfile:
    """Choose dynamic SR tile size, stage eviction, batch sizes, and pixel caps from live hardware."""
    ram_total_mb, ram_avail_mb = query_system_ram_mb()
    gpu_name, vram_total_mb, vram_avail_mb = query_gpu_vram_mb()

    if gpu_name is None or vram_avail_mb <= 0.0:
        # Unknown/no dedicated GPU: conservative defaults + CPU fallback ready
        budget = 0.0
        force_cpu = False
    else:
        budget = min(vram_avail_mb, max(512.0, vram_total_mb * 0.78))
        force_cpu = budget < 900.0

    if os.environ.get("LUMINA_FORCE_CPU", "").strip().lower() in ("1", "true", "yes", "on"):
        force_cpu = True

    if budget >= 7500.0:
        sr_tile, sr_pad, s_batch, evict = 384, 12, 32, False
    elif budget >= 4200.0:
        sr_tile, sr_pad, s_batch, evict = 256, 12, 24, True
    elif budget >= 2200.0:
        sr_tile, sr_pad, s_batch, evict = 192, 12, 16, True
    elif budget > 0.0:
        sr_tile, sr_pad, s_batch, evict = 128, 10, 8, True
    else:
        sr_tile, sr_pad, s_batch, evict = 192, 12, 16, True

    if ram_avail_mb < 6500.0:
        max_work_px, max_zoom_px = 6_000_000, 12_000_000
    elif ram_avail_mb < 11500.0:
        max_work_px, max_zoom_px = 10_000_000, 20_000_000
    else:
        max_work_px, max_zoom_px = 14_000_000, 36_000_000

    if 0.0 < budget < 2400.0:
        max_zoom_px = min(max_zoom_px, 16_000_000)

    return RuntimeProfile(
        gpu_name=gpu_name,
        vram_total_mb=vram_total_mb,
        vram_budget_mb=round(budget, 1),
        ram_total_mb=round(ram_total_mb, 1),
        ram_avail_mb=round(ram_avail_mb, 1),
        sr_tile=sr_tile,
        sr_pad=sr_pad,
        surface_batch=s_batch,
        evict_per_stage=evict,
        force_cpu=force_cpu,
        max_work_px=max_work_px,
        max_zoom_px=max_zoom_px,
    )
