# Exact commands

```powershell
$env:PYTHONPATH="C:\Users\kerok\apps\wd-tagger-trainer"

# recon: meta keys, formats, one-shard timing (writes nothing)
C:\Python314\python.exe scratch\sophie\geom_recon.py

# stage 1 — contiguous sampling -> geom_{train_cache,train_broad,test_cache}.jsonl
C:\Python314\python.exe scratch\sophie\geom_sample.py

# stage 2 — tag forensics + aspect stats + CSVs -> geom_stats.json, *_aspect.csv
C:\Python314\python.exe scratch\sophie\geom_analyze.py

# stage 2b — size-rule details + pixel bar check on 4,000 decoded images
C:\Python314\python.exe scratch\sophie\geom_extra.py

# supporting: exact percentile dumps / corpus tag counts, sampling verification, figure
C:\Python314\python.exe scratch\sophie\geom_numbers.py
C:\Python314\python.exe scratch\sophie\geom_verify.py
C:\Python314\python.exe scratch\sophie\geom_fig.py
```

Sampling is always `_read_samples(tar, start_frac, n)` — `tarfile` iteration from byte 0, so every read is contiguous and no random seeks occur. Headers are read with `PIL.Image.open(io.BytesIO(blob)).size` only (no full decode) except in `geom_extra.py`, where 4000 images are decoded for the pixel bar check.
