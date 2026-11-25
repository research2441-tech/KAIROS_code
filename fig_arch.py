"""Fig. 1: architecture diagram (grayscale, no overlapping elements)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.2})
fig = plt.figure(figsize=(7.16, 3.45), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(-6.5, 46.5); ax.axis("off")


def box(x0, y0, x1, y1, text, fill="white", hatch=None, bold_title=None):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.25,rounding_size=0.9",
                       fc=fill, ec="black", lw=0.8, hatch=hatch)
    ax.add_patch(p)
    if bold_title:
        ax.text((x0 + x1) / 2, y1 - 1.6, bold_title, ha="center", va="top", fontsize=7.4, fontweight="bold")
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 - 1.3, text, ha="center", va="center", fontsize=6.6, linespacing=1.25)
    else:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=6.8, linespacing=1.25)
    return dict(l=x0, r=x1, b=y0, t=y1, cx=(x0 + x1) / 2, cy=(y0 + y1) / 2)


def arrow(pts, dashed=False, lw=0.9, head=True):
    for (xa, ya), (xb, yb) in zip(pts[:-2], pts[1:-1]):
        ax.plot([xa, xb], [ya, yb], color="black", lw=lw, ls=(0, (3.2, 2.0)) if dashed else "-", solid_capstyle="butt")
    a = FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>" if head else "-", mutation_scale=7.5, lw=lw, color="black",
                        linestyle=(0, (3.2, 2.0)) if dashed else "-", shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def label(x, y, s, ha="center", va="center", size=6.3, style="italic"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, style=style,
            bbox=dict(fc="white", ec="none", pad=0.6))


# ---------------- top row: decide before encoding
A1 = box(1.0, 29.5, 12.5, 39.5, "Raw stream\n$\\{I_r\\}_{r=1}^{R}$")
A2 = box(16.5, 29.5, 35.0, 39.5, "$|g_r-g_{r-1}|$ per cell, 56$\\times$56\nspatial median: camera\ntemporal median: persistent", bold_title="Pixel motion probe")
A3 = box(39.0, 29.5, 52.5, 39.5, "$\\hat{E}_{j,n}$ on lattice\n$v_j=\\sum_n \\hat{E}_{j,n}$", bold_title="Velocity field")
A4 = box(56.5, 29.5, 75.0, 39.5, "$p_j\\propto(1-\\rho)\\,v_j^{\\gamma}/\\Sigma+\\rho/T$\nquantiles $\\to j_1<\\dots<j_K$", bold_title="Coverage-floored sampler")
A5 = box(79.0, 29.5, 99.0, 39.5, "", fill="#ececec", hatch="////")
# white patch behind the frozen-encoder text so hatching does not cross it
ax.add_patch(FancyBboxPatch((80.6, 31.0), 16.8, 7.2, boxstyle="round,pad=0.1,rounding_size=0.5", fc="#ececec", ec="none"))
ax.text(A5["cx"], 37.9, "Frozen encoder", ha="center", va="top", fontsize=7.4, fontweight="bold")
ax.text(A5["cx"], A5["cy"] - 1.3, "DINOv2-S/14 on the\n$K$ selected frames only", ha="center", va="center", fontsize=6.6, linespacing=1.25)

# ---------------- bottom row: allocate, encode geometry, reason
B1 = box(73.0, 8.0, 99.0, 18.5, "$s_{k,n}=w_m\\mathcal{Z}(\\mathrm{motion})+w_n\\mathcal{Z}(\\mathrm{novelty})+g_\\phi$\nglobal top-$B$ + mass context tokens", bold_title="Cross-frame token budget")
B2 = box(49.5, 8.0, 69.0, 18.5, "order $\\mathbf{o}_k$ + zero-init gates on\n$\\theta(\\tau_k),\\ \\theta(\\ell_k)$, velocity $\\mathbf{u}_{k,n}$", bold_title="Sampling-geometry encoding")
B3 = box(27.0, 8.0, 45.5, 18.5, "$L=2$ blocks, $d=128$\nproportional attention", bold_title="Spatiotemporal Transformer")
B4 = box(13.0, 8.0, 23.0, 18.5, "Classifier\n$\\hat{\\mathbf{y}}$")
B5 = box(1.0, 8.0, 9.0, 18.5, "Loss\n$\\mathcal{L}$")

# ---------------- forward arrows
arrow([(A1["r"], 34.5), (A2["l"] - 0.4, 34.5)])
arrow([(A2["r"], 34.5), (A3["l"] - 0.4, 34.5)])
arrow([(A3["r"], 34.5), (A4["l"] - 0.4, 34.5)])
arrow([(A4["r"], 34.5), (A5["l"] - 0.4, 34.5)]); label(77.0, 36.6, "$j_k$")
# raw frames fetched for the selected indices only (route above all boxes)
arrow([(A1["cx"], A1["t"]), (A1["cx"], 44.0), (A5["cx"], 44.0), (A5["cx"], A5["t"] + 0.4)])
label(46.0, 44.0, "decode only $I_{c_{j_k}}$ (no other frame reaches the encoder)")
# tokens down to the allocator
arrow([(92.0, A5["b"]), (92.0, B1["t"] + 0.4)]); label(95.6, 24.0, "$\\mathbf{x}_{k,n}$")
# motion energy of the selected frames down to the allocator
arrow([(71.0, A4["b"]), (71.0, 24.0), (80.0, 24.0), (80.0, B1["t"] + 0.4)]); label(75.5, 25.9, "$\\hat{E}_{j_k,n}$")
# timestamps to the geometry encoder
arrow([(60.0, A4["b"]), (60.0, B2["t"] + 0.4)]); label(56.3, 24.0, "$\\tau_k,\\ \\Delta_k$")
# bottom row, right to left
arrow([(B1["l"], 13.25), (B2["r"] + 0.4, 13.25)]); label(71.0, 20.8, "$1+2K+B$ tokens", size=6.0)
arrow([(B2["l"], 13.25), (B3["r"] + 0.4, 13.25)]); label(47.5, 15.3, "$\\mathbf{h}$")
arrow([(B3["l"], 13.25), (B4["r"] + 0.4, 13.25)]); label(25.0, 15.3, "$\\mathbf{z}_0$")
arrow([(B4["l"], 13.25), (B5["r"] + 0.4, 13.25)])

# ---------------- reverse-mode gradient (dashed), stops at the straight-through gate
gy = 3.0
arrow([(B5["cx"], B5["b"]), (B5["cx"], gy), (86.0, gy), (86.0, B1["b"] - 0.4)], dashed=True)
for xg in (18.0, 36.25, 59.25):
    arrow([(xg, gy), (xg, B2["b"] - 0.4)], dashed=True)
label(47.6, gy - 0.05, "$\\partial\\mathcal{L}/\\partial(\\theta,\\phi,w_m,w_n,\\alpha_\\tau,\\alpha_u)$ through the straight-through gate $1+\\sigma(s)-\\mathrm{sg}[\\sigma(s)]$", size=6.2)
# stop-gradient marks
for (x, y) in [(A5["cx"] - 7.0, A5["b"] - 2.0), (A4["cx"] - 3.5, A4["b"] - 2.0)]:
    pass
ax.text(A5["l"] + 1.0, A5["b"] - 1.9, "no gradient", ha="left", va="center", fontsize=6.0, style="italic")
ax.text(59.0, A4["b"] - 1.9, "non-differentiable", ha="right", va="center", fontsize=6.0, style="italic")

# ---------------- legend row (separate from all content)
ly = -3.6
ax.plot([3, 9], [ly, ly], color="black", lw=0.9); ax.text(10, ly, "forward data flow", va="center", fontsize=6.4)
ax.plot([30, 36], [ly, ly], color="black", lw=0.9, ls=(0, (3.2, 2.0))); ax.text(37, ly, "reverse-mode gradient", va="center", fontsize=6.4)
ax.add_patch(FancyBboxPatch((60, ly - 1.2), 5, 2.4, boxstyle="round,pad=0.1,rounding_size=0.3", fc="#ececec", ec="black", lw=0.6, hatch="////"))
ax.text(66.5, ly, "frozen, never updated", va="center", fontsize=6.4)
ax.text(99, ly, "upper row: decided before encoding", va="center", ha="right", fontsize=6.4, style="italic")

fig.savefig("figs/fig1_architecture.png", dpi=300)
print("ok")
