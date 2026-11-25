"""Result figures (grayscale, markers/hatches carry identity, legends kept clear of data)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.7,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
S = json.load(open("results/summary.json")); FL = json.load(open("results/flops.json"))
DSN = {"trace": "TRACE-10", "hmdb": "HMDB51", "ssv2": "SSv2 subset"}
LAB = {"TSN-mean": "TSN mean", "Uniform-CLS": "Uniform-CLS", "Dense-CLS": "Dense-CLS", "Uniform-Full": "Uniform-Full",
       "MG-Full": "MG-Full", "Uniform-Random": "Random drop", "Uniform-STTS": "STTS-style", "Uniform-ToMe": "ToMe",
       "Uniform-EViT": "EViT", "Uniform-RLT": "RLT-style", "MG-STTS": "MG + STTS", "Ours": "KAIROS"}
MK = {"TSN-mean": "v", "Uniform-CLS": "^", "Dense-CLS": "D", "Uniform-Full": "s", "MG-Full": "P", "Uniform-Random": "x",
      "Uniform-STTS": "o", "Uniform-ToMe": "<", "Uniform-EViT": ">", "Uniform-RLT": "h", "MG-STTS": "X", "Ours": "*"}


def clean(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)


def gflops(ds, name):
    """Encoder + probe + temporal model GFLOPs per video."""
    d = S["data"][ds]; enc = FL["enc_gflops_per_frame"]; h = FL["heads"]
    K = d["T"] if name == "Dense-CLS" else 8
    head = {"TSN-mean": 0.0}.get(name)
    if head is None:
        key = name if name in h else ("Uniform-STTS" if name in ("Uniform-Random", "MG-STTS") else
                                      "Uniform-Full" if name == "MG-Full" else "Uniform-CLS")
        head = h[key]["gflops"] if name != "Dense-CLS" else h["Dense-CLS"]["gflops"] * (d["T"] / h["Dense-CLS"]["K"])
    probe = d["nraw_mean"] * FL["probe_mflops_per_frame"] / 1e3 if name in ("MG-Full", "MG-STTS", "Ours") else 0.0
    return K * enc + head + probe


def fig_pareto():
    """Dot plot of top-1 accuracy per method and dataset (mean ± sd over seeds); cost is given in Table main."""
    order = list(LAB)[::-1]
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.75), dpi=300, sharey=True)
    fig.subplots_adjust(left=0.165, right=0.99, top=0.9, bottom=0.15, wspace=0.12)
    for ax, ds in zip(axs, ["trace", "hmdb", "ssv2"]):
        R = S["runs"][ds]
        for yi, n in enumerate(order):
            m, s_, _ = R[n]["test"]
            ax.errorbar(100 * m, yi, xerr=100 * s_, fmt=MK[n], color="black", mfc="black" if n == "Ours" else "white",
                        ms=7.5 if n == "Ours" else 4.2, mew=0.8, elinewidth=0.7, capsize=1.6)
        best = max(100 * R[n]["test"][0] for n in order if n != "Ours")
        ax.axvline(best, color="0.6", lw=0.6, ls=":")
        ax.set_title(DSN[ds], fontsize=8.5); ax.set_xlabel("top-1 accuracy (%)")
        ax.grid(True, axis="x", color="0.9", lw=0.5); clean(ax)
    axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([LAB[n] + (" (6× enc.)" if n == "Dense-CLS" else "") for n in order], fontsize=7)
    axs[0].set_ylim(-0.7, len(order) - 0.3)
    fig.savefig("figs/fig4_pareto.png", dpi=300); plt.close(fig)


def fig_ablation():
    cfgs = [f"Abl-A{a}B{b}C{c}" for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    labels = ["base", "+C", "+B", "+B+C", "+A", "+A+C", "+A+B", "full"]
    fig, ax = plt.subplots(figsize=(3.5, 2.35), dpi=300); fig.subplots_adjust(left=0.15, right=0.99, top=0.84, bottom=0.17)
    hatch = {"trace": "", "hmdb": "////", "ssv2": ""}; face = {"trace": "white", "hmdb": "white", "ssv2": "0.35"}
    w = 0.26
    for i, ds in enumerate(["trace", "hmdb", "ssv2"]):
        R = S["runs"][ds]; base = R["Abl-A0B0C0"]["test"][0]
        vals = [(R["Ours" if c == "Abl-A1B1C1" else c]["test"]) for c in cfgs]
        ax.bar(np.arange(8) + (i - 1) * w, [100 * (v[0] - base) for v in vals], w, yerr=[100 * v[1] for v in vals],
               color=face[ds], edgecolor="black", hatch=hatch[ds], lw=0.6, error_kw=dict(lw=0.6, capsize=1.2), label=DSN[ds])
    ax.axhline(0, color="black", lw=0.6); ax.set_xticks(range(8)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r"$\Delta$ accuracy vs. base (pp)"); clean(ax); ax.grid(True, axis="y", color="0.9", lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7, frameon=False)
    fig.savefig("figs/fig5_ablation.png", dpi=300); plt.close(fig)


def fig_sensitivity():
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.95), dpi=300)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.8, bottom=0.25, wspace=0.42)
    st = {"trace": dict(ls="-", marker="o", mfc="black"), "hmdb": dict(ls="--", marker="s", mfc="white")}

    def get(ds, n):
        R = S["runs"][ds]
        if n == "Ours": return R["Ours"]["test"][0] * 100
        return R[n]["test"][0] * 100 if n in R else np.nan
    panels = [("$\\rho$", [0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0], lambda v: "Ours-rho0" if v == 0 else ("Ours" if v == 0.2 else f"Sens-rho{v}")),
              ("$\\gamma$", [1.0, 2.0, 3.0], lambda v: "Ours" if v == 2.0 else f"Sens-gamma{v}"),
              ("$\\beta$", [0.125, 0.25, 0.5, 1.0], lambda v: "Ours" if v == 0.25 else f"Sens-beta{v}"),
              ("$K$", [4, 8, 16], lambda v: "Ours" if v == 8 else f"Sens-K{v}")]
    for ax, (nm, xs, f) in zip(axs, panels):
        for ds in ("trace", "hmdb"):
            ax.plot(xs, [get(ds, f(x)) for x in xs], color="black", lw=0.9, ms=3.8, mew=0.7, **st[ds])
        if nm == "$\\beta$":
            for ds in ("trace", "hmdb"):
                ys = [get(ds, "Uniform-STTS" if x == 0.25 else f"Sens-STTS-beta{x}") for x in xs]
                ax.plot(xs, ys, color="0.55", lw=0.8, ms=3.2, ls=st[ds]["ls"], marker="^", mfc="white")
            ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels(["1/8", "1/4", "1/2", "1"])
        if nm == "$K$":
            for ds in ("trace", "hmdb"):
                ys = [get(ds, "Uniform-Full" if x == 8 else f"Sens-U-Full-K{x}") for x in xs]
                ax.plot(xs, ys, color="0.55", lw=0.8, ms=3.2, ls=st[ds]["ls"], marker="D", mfc="white")
            ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels(["4", "8", "16"])
        ax.set_xlabel(nm); clean(ax); ax.grid(True, color="0.9", lw=0.5)
    axs[0].set_ylabel("top-1 (%)")
    hs = [plt.Line2D([], [], color="black", lw=0.9, ms=3.8, **st["trace"]), plt.Line2D([], [], color="black", lw=0.9, ms=3.8, **st["hmdb"]),
          plt.Line2D([], [], color="0.55", lw=0.8, marker="^", mfc="white", ms=3.4), plt.Line2D([], [], color="0.55", lw=0.8, marker="D", mfc="white", ms=3.4)]
    fig.legend(hs, ["KAIROS, TRACE-10", "KAIROS, HMDB51", "STTS-style at same $\\beta$", "Uniform-Full at same $K$"],
               loc="upper center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.savefig("figs/fig6_sensitivity.png", dpi=300); plt.close(fig)


def fig_robust():
    meths = ["Uniform-CLS", "Uniform-Full", "MG-Full", "Uniform-STTS", "MG-STTS", "Ours"]
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.2), dpi=300)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.8, bottom=0.2, wspace=0.3)
    phis = [1.0, 0.5, 0.25, 0.125]
    for ax, ds in zip(axs[:2], ["hmdb", "ssv2"]):
        R = S["runs"][ds]
        for n in meths:
            ys = [100 * (R[n]["test"][0] if p == 1.0 else R[n][f"dil{p}"][0]) for p in phis]
            ax.plot(range(4), ys, color="black", lw=1.2 if n == "Ours" else 0.8, marker=MK[n], ms=7 if n == "Ours" else 4,
                    mfc="black" if n == "Ours" else "white", mew=0.7, ls="-" if n in ("Ours", "Uniform-Full") else ":")
        ax.set_xticks(range(4)); ax.set_xticklabels(["1", "1/2", "1/4", "1/8"]); ax.set_xlabel("action fraction $\\varphi$ of the stream")
        ax.set_title(f"({'ab'[['hmdb','ssv2'].index(ds)]}) idle-stream dilution, {DSN[ds]}", fontsize=8); clean(ax); ax.grid(True, color="0.9", lw=0.5)
    axs[0].set_ylabel("top-1 (%)")
    ax = axs[2]
    for ds, ls in (("trace", "-"), ("hmdb", "--"), ("ssv2", ":")):
        R = S["runs"][ds]
        for n, mk in (("Uniform-Full", "s"), ("Ours", "*")):
            ys = [100 * R[n][f"K{k}"][0] if k != 8 else 100 * R[n]["test"][0] for k in (4, 8, 12, 16)]
            ax.plot([4, 8, 12, 16], ys, color="black", ls=ls, marker=mk, ms=6 if mk == "*" else 3.6, mfc="black" if mk == "*" else "white", lw=0.8, mew=0.7)
    ax.set_xticks([4, 8, 12, 16]); ax.set_xlabel("frames at test time (trained with $K=8$)"); ax.set_title("(c) frame-count transfer", fontsize=8)
    clean(ax); ax.grid(True, color="0.9", lw=0.5)
    hs = [plt.Line2D([], [], color="black", ls="-" if n in ("Ours", "Uniform-Full") else ":", marker=MK[n], mfc="black" if n == "Ours" else "white",
                     ms=6 if n == "Ours" else 4, lw=0.8) for n in meths]
    hs2 = [plt.Line2D([], [], color="black", ls=l, lw=0.8) for l in ("-", "--", ":")]
    fig.legend(hs + hs2, [LAB[n] for n in meths] + ["(c) TRACE-10", "(c) HMDB51", "(c) SSv2"], loc="upper center", ncol=9, fontsize=6.6,
               frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=0.9, handletextpad=0.3, handlelength=1.8)
    fig.savefig("figs/fig7_robustness.png", dpi=300); plt.close(fig)


def fig_cd():
    F = S["friedman"]; mr = F["mean_rank"]; cd = F["cd"]
    names = sorted(mr, key=lambda n: mr[n]); M = len(names)
    fig, ax = plt.subplots(figsize=(3.5, 1.45), dpi=300); fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.03)
    ax.set_xlim(1 - 0.3, M + 0.3); ax.set_ylim(-0.55 - 0.42 * ((M + 1) // 2 - 1) - 0.35, 1.75); ax.axis("off")
    ax.plot([1, M], [0, 0], color="black", lw=0.8)
    for t in range(1, M + 1):
        ax.plot([t, t], [0, 0.18], color="black", lw=0.6); ax.text(t, 0.3, str(t), ha="center", va="bottom", fontsize=6.5)
    ax.plot([1, 1 + cd], [1.25, 1.25], color="black", lw=1.4); ax.text(1 + cd / 2, 1.35, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=6.5)
    half = (M + 1) // 2
    for i, n in enumerate(names):
        r = mr[n]
        if i < half:
            y = -0.55 - 0.42 * i; ax.plot([r, r, 0.75], [0, y, y], color="black", lw=0.5); ax.text(0.72, y, f"{LAB[n]} ({r:.1f})", ha="right", va="center", fontsize=6.3)
        else:
            j = M - 1 - i; y = -0.55 - 0.42 * j; ax.plot([r, r, M + 0.25], [0, y, y], color="black", lw=0.5); ax.text(M + 0.28, y, f"({r:.1f}) {LAB[n]}", ha="left", va="center", fontsize=6.3)
    # cliques: groups not significantly different from the best
    best = mr[names[0]]
    grp = [n for n in names if mr[n] - best < cd]
    if len(grp) > 1:
        ax.plot([best - 0.05, mr[grp[-1]] + 0.05], [-0.25, -0.25], color="black", lw=2.0)
    ax.set_xlim(-3.8, M + 4.0)
    fig.savefig("figs/fig8_cd.png", dpi=300, bbox_inches="tight", pad_inches=0.05); plt.close(fig)


def fig_perclass():
    import json as js
    classes = ["brush_hair", "cartwheel", "catch", "chew", "clap", "climb", "climb_stairs", "dive", "draw_sword", "dribble",
               "drink", "eat", "fall_floor", "fencing", "flic_flac", "golf", "handstand", "hit", "hug", "jump", "kick",
               "kick_ball", "kiss", "laugh", "pick", "pour", "pullup", "punch", "push", "pushup", "ride_bike", "ride_horse",
               "run", "shake_hands", "shoot_ball", "shoot_bow", "shoot_gun", "sit", "situp", "smile", "smoke", "somersault",
               "stand", "swing_baseball", "sword", "sword_exercise", "talk", "throw", "turn", "walk", "wave"]
    y = np.load("data/hmdb/labels.npy")[np.load("data/hmdb/split.npy") == 2]
    a = np.array(S["runs"]["hmdb"]["Ours"]["correct_mean"]); b = np.array(S["runs"]["hmdb"]["Uniform-Full"]["correct_mean"])
    d = np.array([100 * (a[y == c].mean() - b[y == c].mean()) for c in range(51)]); o = np.argsort(d)
    fig = plt.figure(figsize=(7.16, 2.6), dpi=300)
    ax = fig.add_axes([0.075, 0.37, 0.585, 0.55])
    ax.bar(range(51), d[o], color=["0.3" if v < 0 else "white" for v in d[o]], edgecolor="black", lw=0.5)
    ax.axhline(0, color="black", lw=0.6); ax.set_xticks(range(51)); ax.set_xticklabels([classes[i].replace("_", " ") for i in o], rotation=90, fontsize=5.2)
    ax.set_ylabel("KAIROS − Uniform-Full (pp)", fontsize=7); ax.set_title("(a) HMDB51 per-class difference (mean over seeds)", fontsize=8, loc="left"); clean(ax)
    ax2 = fig.add_axes([0.74, 0.37, 0.25, 0.55])
    tc = ["R", "L", "U", "D", "CW", "CCW", "grow", "shrink", "R slow", "L slow"]
    w = 0.27
    for i, (n, f, h) in enumerate([("Uniform-Full", "white", ""), ("MG-STTS", "white", "////"), ("Ours", "0.25", "")]):
        ax2.bar(np.arange(10) + (i - 1) * w, [100 * v for v in S["runs"]["trace"][n]["per_class"]], w, color=f, hatch=h, edgecolor="black", lw=0.5, label=LAB[n])
    ax2.set_xticks(range(10)); ax2.set_xticklabels(tc, rotation=90, fontsize=6.2); ax2.set_ylabel("top-1 (%)")
    ax2.set_title("(b) TRACE-10 per class", fontsize=8, loc="left"); clean(ax2)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.33), ncol=3, fontsize=6.2, frameon=False, columnspacing=0.8, handlelength=1.2)
    fig.savefig("figs/fig9_perclass.png", dpi=300, bbox_inches="tight", pad_inches=0.05); plt.close(fig)
    return {classes[i]: float(d[i]) for i in o}


if __name__ == "__main__":
    fig_pareto(); fig_ablation(); fig_sensitivity(); fig_robust(); fig_cd(); pc = fig_perclass()
    json.dump(pc, open("results/perclass_hmdb.json", "w"))
    print("figures done")
