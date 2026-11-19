"""Sample process and device VRAM from NVML on the selected CUDA device."""
import ctypes as C


class Memory(C.Structure):
    _fields_ = [("total", C.c_ulonglong), ("free", C.c_ulonglong), ("used", C.c_ulonglong)]


class Process(C.Structure):
    _fields_ = [
        ("pid", C.c_uint),
        ("usedGpuMemory", C.c_ulonglong),
        ("gpuInstanceId", C.c_uint),
        ("computeInstanceId", C.c_uint),
    ]


class Probe:
    def __init__(self):
        self.lib = C.CDLL("libnvidia-ml.so.1")
        self.check(self.lib.nvmlInit_v2())
        self.handle = C.c_void_p()
        self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(0, C.byref(self.handle)))

    @staticmethod
    def check(rc):
        if rc:
            raise RuntimeError(f"NVML error {rc}")

    def sample(self):
        memory = Memory()
        self.check(self.lib.nvmlDeviceGetMemoryInfo(self.handle, C.byref(memory)))
        count = C.c_uint(128)
        processes = (Process * count.value)()
        self.check(self.lib.nvmlDeviceGetComputeRunningProcesses_v2(
            self.handle, C.byref(count), processes))
        return {
            "device_used_bytes": memory.used,
            "device_total_bytes": memory.total,
            "processes": [{"pid": process.pid, "used_bytes": process.usedGpuMemory}
                          for process in processes[:count.value]],
        }
