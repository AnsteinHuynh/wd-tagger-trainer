import json
s = json.load(open('scratch/sophie/geom_stats.json', encoding='utf-8'))
for r in s['patterns']:
    print(f"[{r['group']}] {r['pattern']!r}  imgs: exact={r['img_exact']} token={r['img_token']} "
          f"sub={r['img_sub']} any={r['img_any']} taghits={r['tagimg_total']}")
    print('   exact vocab:', r['vocab_exact'])
    print('   token vocab:', r['vocab_token'][:12])
    print('   sub   vocab:', r['vocab_sub'][:12])
    print('   top tags:', r['top_tags'])
