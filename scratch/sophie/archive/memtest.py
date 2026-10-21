import ctypes, ctypes.wintypes as wt, gc, os, sys, time
import torch

class PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

def mem(tag):
    p = PMC(); p.cb = ctypes.sizeof(PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(p), p.cb)
    print(f"{tag:28s} WS {p.WorkingSetSize/1e9:6.2f} GB  private(pagefile) {p.PagefileUsage/1e9:6.2f} GB")

mem("start")
paths = [rf"I:\.cache\wd-tagger-trainer\train\pad\shard{k:05d}-10000.pt" for k in range(16)]
t0 = time.time()
objs = [torch.load(p, map_location="cpu", mmap=True, weights_only=False) for p in paths]
print("loaded 16 shards in %.3fs" % (time.time() - t0))
mem("after load")

st = objs[0]["images"].untyped_storage()
print("storage type:", type(st), "nbytes", st.nbytes() / 1e9)
print("has filename attr:", hasattr(st, "filename"), getattr(st, "filename", None))

# touch every page
tot = 0
for o in objs:
    tot += int(o["images"][::7].float().sum())  # read a lot
mem("after touching pages")
print("consumed", tot)
