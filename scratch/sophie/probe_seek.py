import io, json, os, tarfile, time
import torch
R = r"X:\unique\datasets\animetimm-danbooru-wdtagger-v4-w640-ws-full\repo"
p = os.path.join(R, "train", "00001.tar")

# (a) header-only full iteration (no data read)
t0 = time.time()
n = 0
nimg = 0
with tarfile.open(p, "r:") as tf:
    for m in tf:
        n += 1
        if m.name.endswith(".webp"):
            nimg += 1
        tf.extractfile  # no data touch
print("full header iteration: %.2fs, members=%d, webp=%d" % (time.time() - t0, n, nimg))

# (b) skip to middle then read 8 images
t0 = time.time()
with tarfile.open(p, "r:") as tf:
    for i, m in enumerate(tf):
        if i > n // 2:
            break
tmid = time.time() - t0
t0 = time.time()
got = 0
with tarfile.open(p, "r:") as tf:
    for i, m in enumerate(tf):
        if i > n // 2 and m.name.endswith(".webp"):
            b = tf.extractfile(m).read()
            got += 1
            if got >= 8:
                break
print("skip-to-middle: %.2fs, then read 8: %.2fs" % (tmid, time.time() - t0))

# disk type
import subprocess
print(subprocess.run(["powershell", "-NoProfile", "-Command",
    "Get-PhysicalDisk | Select-Object DeviceId,MediaType,Size | Format-Table -AutoSize; Get-Volume | Where-Object {$_.DriveLetter -in 'I','X'} | Select-Object DriveLetter,FileSystem | Format-Table -AutoSize"],
    capture_output=True, text=True).stdout)
