"""Fig.: burst hit rate vs event duration (TRACE-10 test) and analytic cost scaling with the number of frames."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import kairos as kz

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8})
b = kz.Bank("data/trace", light=True); meta = json.load(open("data/trace/meta.json"))["metas"]; te = b.ids(2)
SM = [("Uniform", dict(name="uniform"), "s", "white"), ("MGSampler", dict(name="mg"), "^", "white"),
      ("KAIROS", dict(name="vqs"), "o", "black")]
dur = np.array([meta[i]["t1"] - meta[i]["t0"] for i in te])
out = {}
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.9), dpi=300, gridspec_kw=dict(width_ratios=[1, 1]))
fig.subplots_adjust(left=0.14, right=0.98, top=0.78, bottom=0.25, wspace=0.42)
groups = [("fast", dur < 30), ("slow", dur >= 30)]
for name, smp, mk, fc in SM:
    H, two = [], []
    for i in te:
        m = meta[i]; lat = np.array(m["lattice"]); r = lat[kz.select(b, i, 8, smp)]
        ins = (r >= m["t0"]) & (r < m["t1"] + 4); H.append(ins.mean()); two.append(ins.sum() >= 2)
    H, two = np.array(H), np.array(two)
    out[name] = {g: dict(H=float(H[msk].mean()), two=float(two[msk].mean())) for g, msk in groups}
    for ax, key in zip(axs, ["H", "two"]):
        ys = [out[name][g][key] * (100 if key == "two" else 1) for g, _ in groups]
        ax.plot([0, 1], ys, marker=mk, mfc=fc, color="black", lw=0.9, ms=4.5, mew=0.8, label=name)
for ax, yl, tt in zip(axs, ["hit rate $H$", "videos with ≥2 hits (%)"], ["(a) share of frames in event", "(b) ≥2 frames in event"]):
    ax.set_xticks([0, 1]); ax.set_xticklabels(["fast\n(10–16 fr.)", "slow\n(64–96 fr.)"], fontsize=7)
    ax.set_xlim(-0.35, 1.35); ax.set_ylabel(yl); ax.set_title(tt, fontsize=7.5, loc="left")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(True, axis="y", color="0.9", lw=0.5)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.55, 1.0))
fig.savefig("figs/fig_hitdur.png", dpi=300); plt.close(fig)

# analytic cost scaling (FLOPs = 2 x multiply-accumulates)
FL = json.load(open("results/flops.json")); enc = FL["enc_gflops_per_frame"]
Ks = np.array([8, 16, 32, 64, 128]); L, d, N = 12, 768, 16
def head(M): return 2 * L * (12 * M * d ** 2 + 2 * M ** 2 * d) / 1e9
full = head(1 + Ks * (1 + N)); bud = head(1 + 2 * Ks + (0.25 * Ks * N - Ks)); encc = Ks * enc
fig, ax = plt.subplots(figsize=(3.5, 1.95), dpi=300); fig.subplots_adjust(left=0.15, right=0.98, top=0.8, bottom=0.22)
ax.plot(Ks, encc, color="black", lw=0.9, marker="D", mfc="white", ms=4, label="frozen encoder ($K$ passes)")
ax.plot(Ks, full, color="black", lw=0.9, ls="--", marker="s", mfc="white", ms=4, label="joint attention, all tokens")
ax.plot(Ks, bud, color="black", lw=1.1, marker="o", ms=4, label=r"clip-level budget, $\beta=1/4$")
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(Ks); ax.set_xticklabels([str(k) for k in Ks])
ax.set_xlabel("frames $K$"); ax.set_ylabel("GFLOPs per video")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.grid(True, color="0.9", lw=0.5)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.6, frameon=False, columnspacing=1.0)
fig.savefig("figs/fig_scaling.png", dpi=300); plt.close(fig)
out["scaling"] = dict(K=Ks.tolist(), enc=encc.tolist(), full=full.tolist(), budget=bud.tolist())
json.dump(out, open("results/extra.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:900])
