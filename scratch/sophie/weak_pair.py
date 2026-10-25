import json, numpy as np
CACHE = r"I:\.cache\wd-tagger-trainer\valid_nested"; PD = r"I:\.cache\wd-tagger-trainer\probs"
THR = np.float32(0.40); N = 50000
z = np.load(f"{CACHE}\\labels.npz"); idx, off = z["indices"], z["offsets"]
n_lab = json.load(open(f"{CACHE}\\meta.json"))["n_labels"]
y = np.zeros((N, n_lab), dtype=bool)
for i in range(N): y[i, idx[off[i]:off[i+1]]] = True
def counts(prob):
    tp=np.zeros(n_lab);fp=np.zeros(n_lab);fn=np.zeros(n_lab)
    for s in range(0,N,2000):
        p=prob[s:s+2000]>=THR; yy=y[s:s+2000]
        tp+=(p&yy).sum(0);fp+=(p&~yy).sum(0);fn+=(~p&yy).sum(0)
    return tp,fp,fn
def macro(tp,fp,fn):
    den=2*tp+fp+fn; sup=den>0
    return float((2*tp[sup]/den[sup]).mean())
a=np.asarray(np.load(f"{PD}\\459e2a02_008083.npy", mmap_mode="r")[:N], dtype=np.float32)
c=np.asarray(np.load(f"{PD}\\2b6b81aa_012500.npy", mmap_mode="r")[:N], dtype=np.float32)
print(f"ours 8083            {macro(*counts(a)):.4f}")
print(f"champion             {macro(*counts(c)):.4f}")
best=None
for w in (0.0,0.3,0.4,0.5,0.6,0.7,1.0):
    m=macro(*counts(w*a+(1-w)*c))
    print(f"blend ours {w:.1f}       {m:.4f}")
    tp,fp,fn=counts(w*a+(1-w)*c)
    den=2*tp+fp+fn; sup=den>0
    v=np.zeros(n_lab); v[sup]=2*tp[sup]/den[sup]
    best=v if best is None else np.maximum(best,v)
print(f"per-tag oracle, two weak members: {best[best>0].mean():.4f}")
