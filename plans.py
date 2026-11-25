"""Method definitions and experiment plans."""
from kairos import NP

U = dict(name="uniform"); MG = dict(name="mg"); TOPK = dict(name="topk")


def VQS(rho=.2, gamma=2.0, comp=True):
    return dict(name="vqs", rho=rho, gamma=gamma, comp=comp)


def budget(red, K=8, beta=.25):
    b = int(round(beta * K * NP))
    return b - K if red in ("evit", "ours") else b


def m(sampler, red, K=8, beta=.25, **kw):
    c = dict(sampler=sampler, reducer=red, K=K, **kw)
    if red not in ("mean", "cls", "full"):
        c["B"] = budget(red, K, beta)
    return c


OURS = dict(enc="ctg", ctx=True, prior=True, glob=True, vel=True)
METHODS = {
    "TSN-mean": m(U, "mean"),
    "Uniform-CLS": m(U, "cls"),
    "Dense-CLS": m(U, "cls", K="T"),
    "Uniform-Full": m(U, "full"),
    "MG-Full": m(MG, "full"),
    "Uniform-Random": m(U, "random"),
    "Uniform-STTS": m(U, "stts"),
    "Uniform-ToMe": m(U, "tome"),
    "Uniform-EViT": m(U, "evit"),
    "Uniform-RLT": m(U, "rlt"),
    "MG-STTS": m(MG, "stts"),
    "Ours": m(VQS(), "ours", **OURS),
}
# factorial ablation over A (sampler), B (cross-frame velocity allocation), C (sampling-geometry encoding)
for a in (0, 1):
    for bb in (0, 1):
        for c in (0, 1):
            nm = f"Abl-A{a}B{bb}C{c}"
            METHODS[nm] = m(VQS() if a else U, "ours", enc="ctg" if c else "index", ctx=True,
                            prior=bool(bb), glob=bool(bb), vel=True)
# component-level removals
METHODS["Ours-noComp"] = m(VQS(comp=False), "ours", **OURS)
METHODS["Ours-rho0"] = m(VQS(rho=0.0), "ours", **OURS)
METHODS["Ours-noCtx"] = m(VQS(), "ours", **{**OURS, "ctx": False})
METHODS["Ours-noVel"] = m(VQS(), "ours", **{**OURS, "vel": False})
METHODS["Ours-ungated"] = m(VQS(), "ours", **{**OURS, "enc": "ct"})
# sampler swap under the full proposed head
METHODS["Samp-Uniform"] = m(U, "ours", **OURS)
METHODS["Samp-TopK"] = m(TOPK, "ours", **OURS)
METHODS["Samp-MG"] = m(MG, "ours", **OURS)
# sensitivity
for r in (0.05, 0.1, 0.4, 0.7, 1.0):
    METHODS[f"Sens-rho{r}"] = m(VQS(rho=r), "ours", **OURS)
for g in (1.0, 3.0):
    METHODS[f"Sens-gamma{g}"] = m(VQS(gamma=g), "ours", **OURS)
for be in (0.125, 0.5, 1.0):
    METHODS[f"Sens-beta{be}"] = m(VQS(), "ours", beta=be, **OURS)
    METHODS[f"Sens-STTS-beta{be}"] = m(U, "stts", beta=be)
for k in (4, 16):
    METHODS[f"Sens-K{k}"] = m(VQS(), "ours", K=k, **OURS)
    METHODS[f"Sens-U-Full-K{k}"] = m(U, "full", K=k)

DS = ["trace", "hmdb", "ssv2"]
STRESS = [("dilution", .5), ("dilution", .25), ("dilution", .125), ("noise", .5), ("noise", 1.0),
          ("K", 4), ("K", 12), ("K", 16)]
MAIN = ["TSN-mean", "Uniform-CLS", "Dense-CLS", "Uniform-Full", "MG-Full", "Uniform-Random", "Uniform-STTS",
        "Uniform-ToMe", "Uniform-EViT", "Uniform-RLT", "MG-STTS", "Ours"]
ABL = [f"Abl-A{a}B{b}C{c}" for a in (0, 1) for b in (0, 1) for c in (0, 1)]
COMP = ["Ours-noComp", "Ours-rho0", "Ours-noCtx", "Ours-noVel", "Ours-ungated", "Samp-TopK", "Samp-MG"]
SENS = [k for k in METHODS if k.startswith("Sens-")]


def _jobs(names, dss, seeds, extra=False, epochs=20):
    J = []
    for ds in dss:
        for s in seeds:
            for n in names:
                if n == "Abl-A1B1C1":   # identical to Ours
                    continue
                j = dict(ds=ds, name=n, seed=s, cfg=METHODS[n], epochs=epochs)
                if extra and n in MAIN: j["extra"] = STRESS
                if n in ("Ours", "Uniform-STTS") and s == 0: j["save_model"] = True
                J.append(j)
    return J


PLANS = {
    "pilot": lambda: _jobs(["Uniform-CLS", "Uniform-STTS", "Ours"], ["trace"], [0], epochs=20),
    "main": lambda: _jobs(MAIN, DS, [0, 1, 2], extra=True),
    "abl": lambda: _jobs(ABL, DS, [0, 1, 2]) + _jobs(COMP, ["trace", "hmdb"], [0, 1]) + _jobs(COMP, ["ssv2"], [0]),
    "sens": lambda: _jobs(SENS, ["trace", "hmdb"], [0]),
}
