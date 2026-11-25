"""Aggregate results/ into summary.json used by tables, figures and text."""
import json, os, glob, numpy as np
from scipy import stats
import kairos as kz
from plans import METHODS, MAIN, ABL, SENS, DS

rng = np.random.default_rng(0)
S = {"runs": {}, "hit": {}, "stats": {}, "data": {}}


def ece(P, y, Q=15):
    c = P.max(1); a = (P.argmax(1) == y); e = 0
    edges = np.linspace(0, 1, Q + 1)
    for q in range(Q):
        m = (c > edges[q]) & (c <= edges[q + 1])
        if m.any(): e += m.mean() * abs(a[m].mean() - c[m].mean())
    return float(e)


def macro_f1(pred, y, C):
    f = []
    for c in range(C):
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


banks = {ds: kz.Bank(f"data/{ds}", light=True) for ds in DS if os.path.exists(f"data/{ds}/labels.npy")}
for ds, b in banks.items():
    y = b.y[b.ids(2)]
    S["data"][ds] = dict(N=int(b.N), T=int(b.T), C=int(b.C), tr=int((b.split == 0).sum()), va=int((b.split == 1).sum()),
                         te=int((b.split == 2).sum()), nraw_mean=float(np.load(f"data/{ds}/nraw.npy").mean()),
                         pca_evr=json.load(open(f"data/{ds}/meta.json"))["pca_evr"])
    for d in sorted(glob.glob(f"results/{ds}/*")):
        name = os.path.basename(d); R = []
        for sd in sorted(glob.glob(d + "/seed*/done.json")):
            r = json.load(open(sd)); P = np.load(os.path.dirname(sd) + "/test_probs.npy").astype(np.float32)
            r["f1"] = macro_f1(P.argmax(1), y, b.C); r["ece"] = ece(P, y)
            r["nll"] = float(-np.log(P[np.arange(len(y)), y] + 1e-8).mean())
            r["top5"] = float((np.argsort(-P, 1)[:, :5] == y[:, None]).any(1).mean())
            r["correct"] = (P.argmax(1) == y).astype(np.int8).tolist()
            R.append(r)
        if not R: continue
        agg = {}
        for k in R[0]:
            if isinstance(R[0][k], float) and all(k in r for r in R):
                v = np.array([r[k] for r in R]); agg[k] = [float(v.mean()), float(v.std(ddof=0)), len(v)]
        agg["correct_mean"] = np.mean([r["correct"] for r in R], 0).tolist()
        agg["correct_seeds"] = [r["correct"] for r in R]
        S["runs"].setdefault(ds, {})[name] = agg

# --------------------------------------------------------- burst hit rate of samplers (TRACE-10 test)
if "trace" in banks:
    b = banks["trace"]; meta = json.load(open("data/trace/meta.json"))["metas"]
    te = b.ids(2); lat_step = 4
    for sname, smp in [("Uniform", dict(name="uniform")), ("Top-K motion", dict(name="topk")), ("MGSampler", dict(name="mg")),
                       ("VQS, no compensation", dict(name="vqs", comp=False)), ("VQS, rho=0", dict(name="vqs", rho=0.0)), ("VQS, gamma=1", dict(name="vqs", gamma=1.0)),
                       ("VQS (KAIROS)", dict(name="vqs"))]:
        H, two, shake_hit, H_shk, H_noshk = [], [], [], [], []
        for i in te:
            m = meta[i]; lat = np.array(m["lattice"]); idx = kz.select(b, i, 8, smp)
            r = lat[idx]
            inside = (r >= m["t0"]) & (r < m["t1"] + lat_step)
            H.append(inside.mean()); two.append(inside.sum() >= 2)
            if m["shake"] >= 0:
                shake_hit.append(((r >= m["shake"]) & (r < m["shake"] + 6 + lat_step)).mean()); H_shk.append(inside.mean())
            else:
                H_noshk.append(inside.mean())
        S["hit"][sname] = dict(H=float(np.mean(H)), two=float(np.mean(two)), shake=float(np.mean(shake_hit)),
                               H_shake=float(np.mean(H_shk)), H_noshake=float(np.mean(H_noshk)))
    # fast vs slow classes / shake vs no-shake breakdown for main methods
    y = b.y[te]; shk = np.array([meta[i]["shake"] >= 0 for i in te])
    for name, agg in S["runs"].get("trace", {}).items():
        c = np.array(agg["correct_mean"])
        agg["acc_shake"] = float(c[shk].mean()); agg["acc_noshake"] = float(c[~shk].mean())
        agg["acc_slow"] = float(c[y >= 8].mean()); agg["acc_fastjab"] = float(c[(y <= 1)].mean())
        agg["per_class"] = [float(c[y == k].mean()) for k in range(10)]

# --------------------------------------------------------- statistics: Ours vs baselines
def boot(a, b_, n=10000):
    d = a - b_; idx = rng.integers(0, len(d), (n, len(d)))
    bs = d[idx].mean(1)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())          # two-sided bootstrap p-value
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(min(1.0, max(p, 1.0 / n)))


def mcnemar(a, b_):
    n01 = int(((a == 1) & (b_ == 0)).sum()); n10 = int(((a == 0) & (b_ == 1)).sum())
    if n01 + n10 == 0: return 1.0
    return float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)


for ds in S["runs"]:
    if "Ours" not in S["runs"][ds]: continue
    o = np.array(S["runs"][ds]["Ours"]["correct_mean"]); ov = (o > 0.5).astype(int)
    res = {}
    for name in MAIN:
        if name == "Ours" or name not in S["runs"][ds]: continue
        c = np.array(S["runs"][ds][name]["correct_mean"])
        d, lo, hi, p = boot(o, c); res[name] = dict(delta=d, lo=lo, hi=hi, p=p, mcnemar=mcnemar(ov, (c > 0.5).astype(int)))
    # Holm
    names = list(res); ps = np.array([res[n]["p"] for n in names]); order = np.argsort(ps); m = len(ps); adj = np.zeros(m)
    run = 0
    for r_, i in enumerate(order):
        run = max(run, min(1, (m - r_) * ps[i])); adj[i] = run
    for n, a in zip(names, adj): res[n]["p_holm"] = float(a)
    S["stats"][ds] = res

# Friedman / Nemenyi / Wilcoxon over dataset-seed blocks
blocks, meths = [], [n for n in MAIN]
for ds in S["runs"]:
    for s in range(3):
        row = []
        for n in meths:
            r = S["runs"][ds].get(n)
            if r is None or len(r["correct_seeds"]) <= s: row = None; break
            row.append(np.mean(r["correct_seeds"][s]))
        if row: blocks.append(row)
if len(blocks) >= 3:
    A = np.array(blocks)
    ranks = np.array([stats.rankdata(-r) for r in A])
    fr = stats.friedmanchisquare(*A.T)
    q = {12: 3.268, 11: 3.219, 10: 3.164}[len(meths)]  # Studentized range / sqrt(2), alpha 0.05
    cd = q * np.sqrt(len(meths) * (len(meths) + 1) / (6 * len(A)))
    wil = {}
    oi = meths.index("Ours")
    for j, n in enumerate(meths):
        if n == "Ours": continue
        try: wil[n] = float(stats.wilcoxon(A[:, oi], A[:, j]).pvalue)
        except ValueError: wil[n] = 1.0
    S["friedman"] = dict(chi2=float(fr.statistic), p=float(fr.pvalue), cd=float(cd), nblocks=len(A),
                         mean_rank={n: float(ranks[:, j].mean()) for j, n in enumerate(meths)}, wilcoxon=wil)

for ds in S["runs"]:
    for n in S["runs"][ds]:
        S["runs"][ds][n].pop("correct_seeds", None)
json.dump(S, open("results/summary.json", "w"))
print("ok", {ds: len(S["runs"][ds]) for ds in S["runs"]})
