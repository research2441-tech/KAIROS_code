"""Fig. 2: probe velocity, compensation and the three samplers on one TRACE-10 video (grayscale)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from synth import render, R
from common import probe_energy, lattice, aggregate
import kairos as kz

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8})
# pick a video with a camera bump well separated from a fast event
for vid in range(20000, 20400):
    fr, m = render(vid, 0)
    if m["shake"] > 0 and abs(m["shake"] - m["t0"]) > 50: break
e, res = probe_energy(fr)
T, K = 48, 8
idx = lattice(R, T)
Eraw, Eres = aggregate(e, idx), aggregate(res, idx)
vraw, vres = Eraw.sum(1), Eres.sum(1)
samp = {"Uniform": kz.uniform_sample(T, K), "MGSampler": kz.quantile_sample(vraw, K, 0.0, 1.0),
        r"KAIROS ($\rho=0.2$)": kz.quantile_sample(kz.tmed(Eres).sum(1), K, 0.2, 2.0)}

fig = plt.figure(figsize=(3.5, 4.25), dpi=300)
gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.0, 0.62], hspace=0.62, left=0.13, right=0.97, top=0.97, bottom=0.04)
ax = fig.add_subplot(gs[0])
r = np.arange(R)
ax.plot(r, e.sum(1) / e.sum(1).max(), color="0.6", lw=0.9, label="raw difference energy")
ax.plot(r, res.sum(1) / e.sum(1).max(), color="black", lw=1.0, label="camera-compensated residual")
ax.axvspan(m["t0"], m["t1"], color="0.85", lw=0)
ax.text((m["t0"] + m["t1"]) / 2, 1.08, "event", ha="center", va="bottom", fontsize=7)
ax.annotate("camera bump", xy=(m["shake"] + 3, 0.92), xytext=(m["shake"] + 3 + (30 if m["shake"] < 120 else -60), 0.62),
            fontsize=7, arrowprops=dict(arrowstyle="-|>", lw=0.6, color="black"))
ax.set_xlim(0, R - 1); ax.set_ylim(0, 1.25); ax.set_xlabel("raw frame index $r$"); ax.set_ylabel("normalized energy")
ax.legend(loc="upper left", fontsize=6.5, frameon=False, bbox_to_anchor=(0.0, 1.0), handlelength=1.6)
ax.set_title("(a) probe output", fontsize=8, loc="left", pad=3)

ax = fig.add_subplot(gs[1])
vt = kz.tmed(Eres).sum(1) ** 2; p = 0.8 * vt / vt.sum() + 0.2 / T
ax.step(np.arange(T), np.cumsum(p), where="post", color="black", lw=1.0, label=r"$F_j$ (KAIROS)")
ax.step(np.arange(T), np.cumsum(vraw / vraw.sum()), where="post", color="0.55", lw=0.9, ls="--", label="cumulative raw motion")
ev = np.where((idx >= m["t0"]) & (idx < m["t1"] + 4))[0]
ax.axvspan(ev.min() - 0.5, ev.max() + 0.5, color="0.85", lw=0)
for q in (np.arange(K) + .5) / K:
    ax.axhline(q, color="0.8", lw=0.4, zorder=0)
ax.set_xlim(-0.5, T - 0.5); ax.set_ylim(0, 1.02); ax.set_xlabel("lattice position $j$"); ax.set_ylabel("cumulative mass")
ax.legend(loc="lower right", fontsize=6.5, frameon=False)
ax.set_title("(b) cumulative distributions and quantile targets", fontsize=8, loc="left", pad=3)

ax = fig.add_subplot(gs[2])
marks = ["|", "x", "o"]
for row, ((name, s), mk) in enumerate(zip(samp.items(), marks)):
    yv = 2 - row
    ax.plot(s, [yv] * len(s), ls="none", marker=mk, color="black", ms=5 if mk != "o" else 3.6, mfc="none" if mk == "o" else None, mew=0.9)
    hits = int(((idx[s] >= m["t0"]) & (idx[s] < m["t1"] + 4)).sum())
    ax.text(T + 0.6, yv, f"{hits}/8", va="center", fontsize=7)
ax.axvspan(ev.min() - 0.5, ev.max() + 0.5, color="0.85", lw=0)
ax.set_yticks([2, 1, 0]); ax.set_yticklabels(list(samp.keys()), fontsize=7)
ax.set_xlim(-0.5, T + 4.5); ax.set_ylim(-0.6, 2.6); ax.set_xlabel("lattice position $j$ (right: samples inside the event)")
ax.set_title("(c) selected frames", fontsize=8, loc="left", pad=3)
for a in fig.axes:
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.savefig("figs/fig2_sampler.png", dpi=300)
print(vid, m, {k: v.tolist() for k, v in samp.items()})
