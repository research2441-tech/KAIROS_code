"""Result tables generated from results/summary.json and results/flops.json."""
import json, numpy as np

S = json.load(open("results/summary.json")); FL = json.load(open("results/flops.json"))
DS = ["trace", "hmdb", "ssv2"]
DSN = {"trace": "TRACE-10", "hmdb": "HMDB51", "ssv2": "SSv2"}
LAB = {"TSN-mean": "TSN mean [@tsn]", "Uniform-CLS": "Uniform-CLS", "Dense-CLS": "Dense-CLS", "Uniform-Full": "Uniform-Full [@timesformer]",
       "MG-Full": "MG-Full [@mgsampler]", "Uniform-Random": "Random drop", "Uniform-STTS": "STTS-style [@stts]", "Uniform-ToMe": "ToMe [@tome]",
       "Uniform-EViT": "EViT [@evit]", "Uniform-RLT": "RLT-style [@rlt]", "MG-STTS": "MG + STTS", "Ours": "**KAIROS**"}
MAIN = list(LAB)
TOK = {"TSN-mean": "8", "Uniform-CLS": "9", "Dense-CLS": "T+1", "Uniform-Full": "137", "MG-Full": "137"}


def pm(r, k="test", d=1, scale=100):
    m, s, _ = r[k]; return f"{scale * m:.{d}f} ± {scale * s:.{d}f}"


def gfl(ds, name):
    from figs_results import gflops
    return gflops(ds, name)


def main_table():
    rows = [["Method", "Input tokens", "Encoder calls", "GFLOPs (HMDB51)", "TRACE-10 top-1", "HMDB51 top-1", "HMDB51 macro-F1", "SSv2 top-1", "SSv2 top-5"]]
    best = {ds: max(S["runs"][ds][n]["test"][0] for n in MAIN if n in S["runs"][ds]) for ds in DS}
    for n in MAIN:
        tk = TOK.get(n, "41")
        if n == "Dense-CLS": tk = "49 / 33 / 25"
        enc = "48 / 32 / 24" if n == "Dense-CLS" else "8"
        r = [LAB[n], tk, enc, f"{gfl('hmdb', n):.1f}"]
        for ds, k in (("trace", "test"), ("hmdb", "test"), ("hmdb", "f1"), ("ssv2", "test"), ("ssv2", "top5")):
            v = pm(S["runs"][ds][n], k)
            if k == "test" and abs(S["runs"][ds][n]["test"][0] - best[ds]) < 1e-9: v = "**" + v + "**"
            r.append(v)
        rows.append(r)
    return rows


def hit_table():
    rows = [["Sampler", "$H$", "≥2 in event", "$H$, bump videos", "share on bump"]]
    for k, v in S["hit"].items():
        rows.append([k.replace("rho", "$\\rho$").replace("gamma", "$\\gamma$"), f"{v['H']:.3f}", f"{100 * v['two']:.1f}%", f"{v['H_shake']:.3f}", f"{v['shake']:.3f}"])
    return rows


def abl_table():
    rows = [["A", "B", "C", "TRACE-10", "HMDB51", "SSv2"]]
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                n = "Ours" if (a, b, c) == (1, 1, 1) else f"Abl-A{a}B{b}C{c}"
                rows.append(["✓" if a else "–", "✓" if b else "–", "✓" if c else "–"] + [pm(S["runs"][ds][n]) for ds in DS])
    return rows


COMPN = {"Ours": "KAIROS (full)", "Ours-noComp": "no camera compensation", "Ours-rho0": "no coverage floor ($\\rho=0$)",
         "Ours-noCtx": "no context tokens", "Ours-noVel": "no feature velocity $\\mathbf{u}_{k,n}$",
         "Samp-MG": "sampler swapped for MGSampler", "Samp-TopK": "sampler swapped for top-$K$ motion",
         "Abl-A0B1C1": "sampler swapped for uniform"}


def comp_table():
    rows = [["Variant", "TRACE-10", "HMDB51", "SSv2"]]
    for n, lab in COMPN.items():
        rows.append([lab] + [pm(S["runs"][ds][n]) for ds in DS])
    return rows


def robust_table():
    meths = ["Uniform-CLS", "Uniform-Full", "MG-Full", "Uniform-STTS", "MG-STTS", "Ours"]
    rows = [["Method", ("HMDB51 dilution $\\varphi$", 3), ("SSv2 dilution $\\varphi$", 3), ("probe noise (HMDB51)", 2), ("$K=16$ at test", 3)],
            ["", "1/2", "1/4", "1/8", "1/2", "1/4", "1/8", "$\\sigma=0.5$", "$\\sigma=1$", "TRACE", "HMDB", "SSv2"]]
    for n in meths:
        r = [LAB[n].replace("**", "")]
        for ds in ("hmdb", "ssv2"):
            for p in (0.5, 0.25, 0.125): r.append(f"{100 * S['runs'][ds][n][f'dil{p}'][0]:.1f}")
        for s in (0.5, 1.0): r.append(f"{100 * S['runs']['hmdb'][n][f'noise{s}'][0]:.1f}")
        for ds in DS: r.append(f"{100 * S['runs'][ds][n]['K16'][0]:.1f}")
        rows.append(r)
    return rows


def eff_table():
    h = FL["heads"]
    rows = [["Method", "Encoder calls", "Head tokens", "Head GFLOPs", "Total GFLOPs", "Head latency (ms)", "Trainable params"]]
    for n, key in (("Uniform-CLS", "Uniform-CLS"), ("Dense-CLS", "Dense-CLS"), ("Uniform-Full", "Uniform-Full"), ("Uniform-STTS", "Uniform-STTS"),
                   ("Uniform-ToMe", "Uniform-ToMe"), ("Uniform-EViT", "Uniform-EViT"), ("Uniform-RLT", "Uniform-RLT"), ("Ours", "Ours")):
        x = h[key]; ntok = {"Dense-CLS": "33", "Uniform-CLS": "9", "Uniform-Full": "137"}.get(n, "41")
        rows.append([LAB[n].replace("**", ""), "32" if n == "Dense-CLS" else "8", ntok, f"{x['gflops']:.3f}", f"{gfl('hmdb', n):.2f}",
                     f"{x['ms']:.2f}", f"{x['params'] / 1e6:.3f} M"])
    return rows


def calib_table():
    meths = ["Uniform-CLS", "Uniform-Full", "MG-Full", "Uniform-STTS", "MG-STTS", "Ours"]
    rows = [["Method", ("TRACE-10", 2), ("HMDB51", 2), ("SSv2", 2)], ["", "ECE (%)", "NLL", "ECE (%)", "NLL", "ECE (%)", "NLL"]]
    for n in meths:
        r = [LAB[n].replace("**", "")]
        for ds in DS:
            r += [f"{100 * S['runs'][ds][n]['ece'][0]:.1f}", f"{S['runs'][ds][n]['nll'][0]:.2f}"]
        rows.append(r)
    return rows


def stats_table():
    rows = [["Baseline", ("TRACE-10", 2), ("HMDB51", 2), ("SSv2", 2), "Wilcoxon"],
            ["", "Δ [95% CI] (pp)", "$p_{\\mathrm{Holm}}$", "Δ [95% CI] (pp)", "$p_{\\mathrm{Holm}}$", "Δ [95% CI] (pp)", "$p_{\\mathrm{Holm}}$", "$p$"]]
    def fp(p): return "<0.001" if p < 1e-3 else f"{p:.3f}"
    for n in MAIN:
        if n == "Ours": continue
        r = [LAB[n].replace("**", "")]
        for ds in DS:
            st = S["stats"][ds][n]
            r += [f"{100 * st['delta']:+.1f} [{100 * st['lo']:+.1f}, {100 * st['hi']:+.1f}]", fp(st["p_holm"])]
        r.append(fp(S["friedman"]["wilcoxon"][n]))
        rows.append(r)
    return rows


def data_table():
    d = S["data"]
    rows = [["Property", "TRACE-10", "HMDB51 [@hmdb]", "SSv2 subset [@ssv2]"],
            ["Classes", "10", "51", "174"],
            ["Train / val / test", f"{d['trace']['tr']} / {d['trace']['va']} / {d['trace']['te']}", f"{d['hmdb']['tr']} / {d['hmdb']['va']} / {d['hmdb']['te']}",
             f"{d['ssv2']['tr']} / {d['ssv2']['va']} / {d['ssv2']['te']}"],
            ["Raw frames per clip (mean)", f"{d['trace']['nraw_mean']:.0f}", f"{d['hmdb']['nraw_mean']:.0f}", f"{d['ssv2']['nraw_mean']:.0f}"],
            ["Lattice $T$", "48", "32", "24"],
            ["Label carried by", "5–8% of frames (fast classes)", "appearance and motion", "motion and interaction"],
            ["Partition rule", "disjoint seeds", "official split 1, group-aware val.", "stratified 70/10/20"]]
    return rows
