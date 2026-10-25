import time

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


from wdtag.labels import load_label_space
from wdtag.stream import TarStream

n_labels = len(load_label_space())
log("stream only, no model")
st = TarStream("train", seed=1337, pass_index=0, batch_size=64, n_labels=n_labels,
               workers=12, prefetch=8, max_samples=20000)
log("open")
n = 0
t1 = time.time()
for item in st.batches():
    images = item[0]
    n += 1
    if n <= 3 or n % 20 == 0:
        shape = tuple(images.shape) if hasattr(images, "shape") else type(images).__name__
        log(f"batch {n} {shape} {n * 64 / (time.time() - t1):.0f} img/s")
    if n >= 60:
        break
log("done")
