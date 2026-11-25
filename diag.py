import numpy as np, torch, json, sys
torch.set_num_threads(1)
import kairos as kz
b = kz.Bank("data/trace"); meta = json.load(open("data/trace/meta.json"))["metas"]
orig = kz.select
def oracle(bank, i, K, sampler, rng=None, Eres=None, Eraw=None, T=None):
    if sampler["name"] != "oracle": return orig(bank, i, K, sampler, rng, Eres, Eraw, T)
    m = meta[i]; lat = np.array(m["lattice"])
    a = np.searchsorted(lat, m["t0"]) - 1; z = np.searchsorted(lat, m["t1"]) + 1
    idx = np.round(np.linspace(max(a,0), min(z, b.T-1), K)).astype(int)
    return kz._strict(idx, b.T)
kz.select = oracle
for red in sys.argv[2:]:
    cfg = dict(sampler=dict(name="oracle"), reducer=red, enc="ct")
    m, val, secs = kz.train(b, cfg, seed=0, epochs=int(sys.argv[1]), K=8)
    te = b.ids(2); P = kz.predict(m, b, te, 8, cfg["sampler"])
    print("oracle", red, f"val {val:.3f} test {(P.argmax(1)==b.y[te]).mean():.3f} {secs:.0f}s", flush=True)
