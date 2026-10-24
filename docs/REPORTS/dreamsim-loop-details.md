# Loop-level details

Three `.item()` per step — a real (small) sync tax: `loss_sum.item()` (`train.py:1608`), `correct.item()` / `scored.item()` (`1618-1619`), plus a `.cpu()` per preview (`1443-1452`, `1566-1580`). Each drains the stream, blocking overlap of next H2D and optimizer. → Low-medium (~1–2 ms of 359 ms): accumulate on device, read once per log tick
Non-finite guard is a per-step scalar reduction (`torch.isfinite(loss_sum)`, `train.py:1593`); expensive per-input diagnostic only on failure (`1594-1599`). → Copy the shape: fail loudly, name the input, keep the last checkpoint usable
Previews budget-bounded, read once per step (`train.py:1566-1580`, `preview_images=10`). Maps to your first-100-PNG requirement — countdown + batch the `.cpu()` copies
No stream usage; H2D is plain `non_blocking` into fresh tensors (`train.py:1430-1431`, `1559-1560`). → Gap: double-buffered `next_batch` prefetch on a copy stream is the textbook next step
Resume without re-decoding at sampler level (`data.py:841-844`) or via pre-sliced indices (`train.py:1380-1386`: "iterating N pre-sliced batches (consumed prefix never decoded)")
Hot-loop bookkeeping is plain ints/floats — no dict lookups, no `time.time()` outside the log branch; their per-step Python is ~0 (contrast section 5)
