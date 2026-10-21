import csv, json
S = json.load(open('scratch/sophie/geom_stats.json', encoding='utf-8'))
for s in S['stats']:
    print('==', s['name'])
    for k in ('aspect_pct', 'h_pct', 'd_pct', 'waste_pct'):
        print(' ', k, {q: round(v, 4) for q, v in s[k].items()})
    print('  buckets', [(b['lo'], b['hi'], b['n'], round(b['pct'], 2)) for b in s['absdev_buckets']])
    print('  common', [(c['label'], c['n'], round(c['pct'], 2)) for c in s['common']])
    print('  orient', s['orientation'])
    print('  dfrac', {k: (v['n'], round(v['pct'], 2)) for k, v in s['d_frac'].items()})
    print('  mean waste', round(100 * s['waste_mean'], 2))

want = ['white_background', 'simple_background', 'letterboxed', 'pillarboxed', 'border', 'white_border',
        'outside_border', 'black_border', 'grey_border', 'letterboxed', 'deformed', 'bad_anatomy',
        'bad_proportions', 'distortion', 'comic', '4koma', '2koma', 'multiple_views', 'split_screen',
        'out_of_frame', 'borderless_panels', 'multiple_4koma', 'stretching']
cnt = {}
with open(r'X:\unique\datasets\animetimm-danbooru-wdtagger-v4-w640-ws-full\repo\tags.csv', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        if r['name'] in want:
            cnt[r['name']] = (int(r['count']), int(r['train_count']), int(r['test_count']))
TOT = 5914596
print('\ntag total / train / test / pct of corpus')
for w in want:
    if w in cnt:
        c = cnt[w]
        print(f'  {w:20s} {c[0]:9,d} {c[1]:9,d} {c[2]:8,d}   {100*c[0]/TOT:6.3f}%')
print('corpus total images', TOT)
