"""One purpose: does the streamer keep its order when the consumer pulls in bursts?

The accumulation loop takes `grad_accum` batches back to back before touching the GPU, which is
a different pull rhythm from one batch per optimiser step. If the ring buffer can hand back a
slot that the producer has already overwritten, the sequence changes under that rhythm and no
two runs agree. Prints one hash per variant; run it twice and compare the two lines.

    python scratch/sophie/stream_burst_probe.py
"""

import hashlib
import time

from wdtag.stream import TarStream

N = 300


def run(burst: int, gap: float) -> str:
    sp = TarStream("train", seed=1337, pass_index=0, batch_size=4, n_labels=12476,
                   workers=8, prefetch=4, skip=0, max_samples=10 ** 9, log=lambda m: None)
    h = hashlib.sha256()
    n = 0
    for x, y, consumed, src in sp.batches():
        h.update(str(tuple(src)).encode())
        h.update(x.numpy().tobytes())
        n += 1
        if n % burst == 0:
            time.sleep(gap)
        if n >= N:
            break
    return h.hexdigest()[:16]


if __name__ == "__main__":
    print(f"burst 1 gap 0    {run(1, 0.0)}")
    print(f"burst 3 gap 0.02 {run(3, 0.02)}")
    print(f"burst 1 gap 0    {run(1, 0.0)}")
