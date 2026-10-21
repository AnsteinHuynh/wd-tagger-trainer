import sys, time, torch
paths = [rf"I:\.cache\wd-tagger-trainer\train\pad\shard{k:05d}-10000.pt" for k in range(16)]
objs = [torch.load(p, map_location="cpu", mmap=True, weights_only=False) for p in paths]
print("loaded", flush=True)
tot = 0.0
for o in objs:
    tot += float(o["images"].float().mean())
print("touched all pages", flush=True)
for i in range(6):
    time.sleep(2)
    print(f"alive {i}", flush=True)
