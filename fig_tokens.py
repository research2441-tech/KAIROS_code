"""Fig. 3: frames and retained regions for one TRACE-10 test video (grayscale thumbnails)."""
import json, numpy as np, torch, matplotlib, cv2
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import kairos as kz
from plans import METHODS
from synth import render, CLASSES

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.5})
b = kz.Bank("data/trace"); meta = json.load(open("data/trace/meta.json"))["metas"]
te = b.ids(2)
models = {}
for name in ("Uniform-STTS", "Ours"):
    cfg = dict(METHODS[name]); cfg.pop("K")
    m = kz.build(b, cfg); m.load_state_dict(torch.load(f"results/trace/{name}/seed0/model.pt", map_location="cpu"), strict=False); m.eval()
    models[name] = (m, cfg)


def kept_mask(m, cfg, i):
    idx = kz.select(b, i, 8, cfg["sampler"])
    X = torch.from_numpy(b.tok[i, idx].astype(np.float32))[None]; M = torch.from_numpy(b.Etm[i, idx])[None]
    P = X[:, :, 1:]; Prev = torch.cat([P[:, :1], P[:, :-1]], 1); cls_t = X[:, :, 0]
    with torch.no_grad():
        s, _ = m._scores(P, Prev, M, cls_t, cfg["reducer"])
        if cfg["reducer"] == "ours":
            top = s.reshape(1, -1).topk(cfg["B"], 1).indices
            keep = torch.zeros(1, 8 * kz.NP, dtype=torch.bool).scatter_(1, top, True).view(8, kz.NP)
        else:
            top = s.topk(cfg["B"] // 8, -1).indices
            keep = torch.zeros(1, 8, kz.NP, dtype=torch.bool).scatter_(2, top, True)[0]
        pred = m(X, M, torch.from_numpy(idx / (b.T - 1)).float()[None]).argmax(1).item()
    return idx, keep.numpy(), pred


# choose a test video of a fast class where KAIROS is correct and the uniform model is wrong
pick = None
for i in te:
    if b.y[i] in (0, 2, 4) and meta[i]["shake"] < 0:
        a = kept_mask(*models["Ours"], i); u = kept_mask(*models["Uniform-STTS"], i)
        if a[2] == b.y[i] and u[2] != b.y[i]:
            pick = i; break
i = pick if pick is not None else te[0]
fr, mt = render(int(i), int(b.y[i]))
lat = np.array(meta[i]["lattice"])
fig, axs = plt.subplots(2, 8, figsize=(7.16, 2.35), dpi=300)
fig.subplots_adjust(left=0.085, right=0.995, top=0.86, bottom=0.04, wspace=0.06, hspace=0.32)
info = {}
for row, name in enumerate(("Uniform-STTS", "Ours")):
    idx, keep, pred = kept_mask(*models[name], i)
    info[name] = dict(frames=lat[idx].tolist(), kept_per_frame=keep.sum(1).tolist(), pred=CLASSES[pred])
    for k in range(8):
        ax = axs[row, k]; r = int(lat[idx[k]])
        g = cv2.cvtColor(fr[r], cv2.COLOR_RGB2GRAY).astype(np.float32)
        img = np.stack([g] * 3, -1) / 255.0
        for n in range(kz.NP):
            y0, x0 = (n // 4) * 28, (n % 4) * 28
            if not keep[k, n]:
                img[y0:y0 + 28, x0:x0 + 28] = 0.55 * img[y0:y0 + 28, x0:x0 + 28] + 0.45
        ax.imshow(img, interpolation="nearest")
        for n in range(kz.NP):
            if keep[k, n]:
                y0, x0 = (n // 4) * 28, (n % 4) * 28
                ax.add_patch(Rectangle((x0 + 0.5, y0 + 0.5), 27, 27, fill=False, ec="black", lw=0.9))
        inside = mt["t0"] <= r < mt["t1"] + 4
        ax.set_title(f"$r$={r}" + (" (event)" if inside else ""), fontsize=6.5, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
    axs[row, 0].set_ylabel("STTS-style\nuniform" if row == 0 else "KAIROS", fontsize=7.5)
fig.text(0.5, 0.965, f"true class: {CLASSES[b.y[i]]};  STTS-style predicts {info['Uniform-STTS']['pred']},  KAIROS predicts {info['Ours']['pred']}",
         ha="center", va="center", fontsize=7.5)
fig.savefig("figs/fig3_tokens.png", dpi=300)
json.dump(dict(video=int(i), event=[mt["t0"], mt["t1"]], **info), open("results/fig3_info.json", "w"), indent=1)
print(info)
