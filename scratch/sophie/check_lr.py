from wdtag.train import lr_at

for s in (9457, 11000, 15000, 20000, 27718):
    a = lr_at(s, 27718, 1.79e-4, 300, "cosine", 0, 9400)
    b = lr_at(s, 27718, 1.469e-5, 300, "cosine", 0, 9400)
    h = lr_at(s, 27718, 8.231e-5, 300, "cosine", 0, 9400)
    print(s, f"adapter {a:.3e} block {b:.2e} head {h:.2e}")
print("total steps", -(-10643426 // 384))
