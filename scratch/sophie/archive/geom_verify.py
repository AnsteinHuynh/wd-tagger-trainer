"""Verify the replicated sampling against the real cache's own artifacts."""
import csv, glob, json, os
from collections import Counter, OrderedDict

rows_tc = [json.loads(l) for l in open('scratch/sophie/geom_train_cache.jsonl', encoding='utf-8')]
idx = {(r['key'], r['shard']): r['row'] for r in rows_tc}
print('replicated train rows:', len(rows_tc))

allman = glob.glob('runs/*/debugpics/manifest.json')
tot = found = 0
missing = []
for m in allman:
    for e in json.load(open(m, encoding='utf-8')):
        tot += 1
        if (str(e['key']), e['tar']) in idx:
            found += 1
        else:
            missing.append((e['key'], e['tar']))
print(f'debugpics manifests: {len(allman)} runs, {tot} entries, found in replication: {found}, missing: {len(missing)}')
print('example missing:', missing[:5])

# shard -> row ranges in both CSVs
for name in ('test_aspect.csv', 'train_sample_aspect.csv'):
    with open(os.path.join('scratch/sophie', name), encoding='utf-8') as f:
        rr = list(csv.DictReader(f))
    g = OrderedDict()
    for r in rr:
        g.setdefault(r['shard'], []).append(int(r['row']))
    print(f"\n{name}: {len(rr)} rows, {len(g)} shards")
    items = list(g.items())
    for s, rws in items[:4]:
        print(f"   {s}: rows {rws[0]}..{rws[-1]} (n={len(rws)}, contiguous={rws == list(range(rws[0], rws[-1]+1))})")
    print(f"   ... {items[-1][0]}: rows {items[-1][1][0]}..{items[-1][1][-1]} (n={len(items[-1][1])})")
    print('   dup keys:', len(rr) - len({r['key'] for r in rr}))
    print('   row == index:', all(int(r['row']) == i for i, r in enumerate(rr)))
