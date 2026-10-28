"""VRAM held by other programs on one NVIDIA GPU, read through NVML (the driver's libnvidia-ml) with ctypes.

cogito motoris uses it to decide, between requests, whether the expert cache should leave room for other programs
(see Service.vram_mode in server.py).  Nothing here allocates on the GPU or talks to the engine.
"""
from __future__ import annotations

import ctypes
import ctypes.util


class _Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _Process(ctypes.Structure):   # nvmlProcessInfo_t (the _v2/_v3 process queries)
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]


MIB = 1 << 20


class VramWatch:
    """`others_mib(own_pids)`: VRAM in use by processes other than `own_pids`, in MiB; None when NVML is missing."""

    def __init__(self, index: int = 0):
        self.lib = None
        self.dev = ctypes.c_void_p()
        name = ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
        try:
            lib = ctypes.CDLL(name)
            if lib.nvmlInit_v2() != 0 or lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.dev)) != 0:
                return
            self.lib = lib
        except OSError:
            self.lib = None

    def available(self) -> bool:
        return self.lib is not None

    def memory(self) -> tuple[int, int] | None:
        """(total, used) in MiB."""
        if self.lib is None:
            return None
        m = _Memory()
        if self.lib.nvmlDeviceGetMemoryInfo(self.dev, ctypes.byref(m)) != 0:
            return None
        return m.total // MIB, m.used // MIB

    def _processes(self, fn) -> list[tuple[int, int]]:
        n = ctypes.c_uint(0)
        rc = fn(self.dev, ctypes.byref(n), None)
        if rc not in (0, 7) or n.value == 0:   # 7: NVML_ERROR_INSUFFICIENT_SIZE, the count is in n
            return []
        buf = (_Process * (n.value + 8))()
        n = ctypes.c_uint(n.value + 8)
        if fn(self.dev, ctypes.byref(n), buf) != 0:
            return []
        out = []
        for p in buf[:n.value]:
            used = p.usedGpuMemory if p.usedGpuMemory < (1 << 62) else 0   # NVML_VALUE_NOT_AVAILABLE
            out.append((int(p.pid), int(used // MIB)))
        return out

    def others_mib(self, own_pids: set[int]) -> int | None:
        """MiB used by every process but `own_pids`: the card's used memory minus what our processes hold, so memory
        the driver does not attribute to a process (the compositor's share, for one) counts as another program's."""
        mem = self.memory()
        if mem is None:
            return None
        own = 0
        seen = set()
        for fn_name in ("nvmlDeviceGetComputeRunningProcesses_v3", "nvmlDeviceGetGraphicsRunningProcesses_v3"):
            fn = getattr(self.lib, fn_name, None)
            if fn is None:
                continue
            for pid, used in self._processes(fn):
                if pid in own_pids and pid not in seen:
                    own += used
                    seen.add(pid)
        return max(0, mem[1] - own)
