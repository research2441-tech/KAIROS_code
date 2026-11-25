import numpy as np, torch, time, sys
torch.set_num_threads(1)
import kairos as kz
from plans import METHODS
b = kz.Bank("data/trace")
ep=int(sys.argv[1])
for name in sys.argv[2:]:
    cfg = dict(METHODS[name]); K = cfg.pop("K")
    m, val, secs = kz.train(b, cfg, seed=0, epochs=ep, K=K)
    te = b.ids(2); P = kz.predict(m, b, te, K, cfg["sampler"])
    print(name, f"val {val:.3f} test {(P.argmax(1)==b.y[te]).mean():.3f} {secs:.0f}s", flush=True)
