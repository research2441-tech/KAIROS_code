import sys, numpy as np, torch
torch.set_num_threads(1)
sys.argv_saved = sys.argv; ds = sys.argv[1]; names = sys.argv[2:]
sys.argv = ["x", "main", "0", "1"]
import kairos as kz
from plans import METHODS, VQS, m
import importlib.util
spec = importlib.util.spec_from_file_location("rx", "run_exp.py")
src = open("run_exp.py").read().split("for n, job in enumerate(jobs):")[0]
ns = {}; exec(src.replace("jobs = [j for j in PLANS[plan]() if not only or j[\"ds\"] == only]", "jobs = []"), ns)
b = kz.Bank(f"data/{ds}")
V = {"Ours": METHODS["Ours"], "A0B1C1": METHODS["Abl-A0B1C1"], "A1B0C1": METHODS["Abl-A1B0C1"], "A1B1C0": METHODS["Abl-A1B1C0"],
     "Ours-rel": m(VQS(), "ours", enc="rel", ctx=True, prior=True, glob=True, vel=True), "MG-STTS": METHODS["MG-STTS"], "Ours-ctg": m(VQS(), "ours", enc="ctg", ctx=True, prior=True, glob=True, vel=True)}
va = b.ids(1)
for n in names:
    cfg = dict(V[n]); K = cfg.pop("K"); mdl, val, secs = kz.train(b, cfg, seed=0, epochs=20, K=K)
    out = [f"val {val:.4f}"]
    for phi in (0.25, 0.125):
        P = kz.predict(mdl, b, va, K, cfg["sampler"], view=ns["dilution_view"](b, phi))
        out.append(f"valdil{phi} {(P.argmax(1) == b.y[va]).mean():.4f}")
    print(ds, n, *out, f"{secs:.0f}s", flush=True)
