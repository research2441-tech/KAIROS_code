"""Decode clips, run the pixel motion probe on every raw frame, encode a uniform candidate
lattice with frozen DINOv2-S/14, and store PCA-reduced tokens.
usage: python prep.py {trace|hmdb|ssv2} [--limit N]"""
import sys, os, io, json, time, re, numpy as np, torch
import pyarrow.parquet as pq
from common import *

torch.set_num_threads(2)
name = sys.argv[1]
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
LO = int(sys.argv[sys.argv.index("--lo") + 1]) if "--lo" in sys.argv else 0
HI = int(sys.argv[sys.argv.index("--hi") + 1]) if "--hi" in sys.argv else 10 ** 9
out = f"data/{name}"; os.makedirs(out, exist_ok=True)
T = {"trace": 48, "hmdb": 32, "ssv2": 24}[name]
HMDB = ["brush_hair", "cartwheel", "catch", "chew", "clap", "climb", "climb_stairs", "dive", "draw_sword", "dribble",
        "drink", "eat", "fall_floor", "fencing", "flic_flac", "golf", "handstand", "hit", "hug", "jump", "kick",
        "kick_ball", "kiss", "laugh", "pick", "pour", "pullup", "punch", "push", "pushup", "ride_bike", "ride_horse",
        "run", "shake_hands", "shoot_ball", "shoot_bow", "shoot_gun", "sit", "situp", "smile", "smoke", "somersault",
        "stand", "swing_baseball", "sword", "sword_exercise", "talk", "throw", "turn", "walk", "wave"]


def decode(b):
    """Streams the clip once, keeping only 112x112 crops (encoder input) and 56x56 grey probes,
    so that memory stays bounded for very long clips."""
    import av
    crops, grey = [], []
    with av.open(io.BytesIO(b)) as c:
        for f in c.decode(video=0):
            cr = crop_resize(f.to_ndarray(format="rgb24")); crops.append(cr); grey.append(probe_grey(cr))
    return crops, grey


def items():
    """Yields (index, label, split, frames, meta)."""
    if name == "trace":
        from synth import render
        n = [2000, 400, 600]
        lab = np.concatenate([np.random.default_rng(s).permutation(np.arange(k) % 10) for s, k in enumerate(n)])
        spl = np.repeat([0, 1, 2], n)
        for i in range(len(lab)):
            fr, meta = render(i, int(lab[i]))
            yield i, int(lab[i]), int(spl[i]), (list(fr), [probe_grey(f) for f in fr]), meta
        return
    if name == "hmdb":
        files = [("train-0000%d-of-00003" % k, 0) for k in range(3)] + [("test-00000-of-00001", 2)]
        tabs = [(pq.read_table(f"raw/hmdb/{f}.parquet", columns=["label"]), s) for f, s in files]
        paths = []
        for f, s in files:
            pf = pq.ParquetFile(f"raw/hmdb/{f}.parquet")
            for bt in pf.iter_batches(batch_size=16, columns=["video"]):
                paths += [v["path"] for v in bt.column(0).to_pylist()]
        labs = np.concatenate([t[0]["label"].to_numpy() for t in tabs])
        spl = np.concatenate([np.full(len(t), s) for t, s in tabs])
        # group-aware validation split: clips cut from the same source video stay together
        groups = np.array([p.split("_" + HMDB[l] + "_")[0] for p, l in zip(paths, labs)])
        rng = np.random.default_rng(0)
        for c in range(51):
            gs = sorted(set(groups[(labs == c) & (spl == 0)])); rng.shuffle(gs)
            tot, cnt = ((labs == c) & (spl == 0)).sum(), 0
            for g in gs:
                if cnt >= 0.15 * tot: break
                m = (groups == g) & (labs == c) & (spl == 0); spl[m] = 1; cnt += m.sum()
        json.dump(paths, open(f"{out}/paths.json", "w"))
        i = 0
        for f, s in files:
            pf = pq.ParquetFile(f"raw/hmdb/{f}.parquet")
            for bt in pf.iter_batches(batch_size=16, columns=["video"]):   # small batches keep memory bounded
                for v in bt.column(0).to_pylist():
                    if LO <= i < HI:
                        yield i, int(labs[i]), int(spl[i]), decode(v["bytes"]), {}
                    i += 1
                if i >= HI: return
        return
    if name == "ssv2":
        f = "raw/ssv2/test-00000-of-00001.parquet"
        labs = pq.read_table(f, columns=["label"])["label"].to_numpy()
        rng = np.random.default_rng(0); spl = np.zeros(len(labs), np.int64)
        for c in np.unique(labs):
            idx = rng.permutation(np.where(labs == c)[0]); n = len(idx)
            spl[idx[:round(.2 * n)]] = 2; spl[idx[round(.2 * n):round(.3 * n)]] = 1
        pf = pq.ParquetFile(f); i = 0
        for bt in pf.iter_batches(batch_size=16, columns=["video"]):
            for v in bt.column(0).to_pylist():
                yield i, int(labs[i]), int(spl[i]), decode(v["bytes"]), {}
                i += 1


N = {"trace": 3000, "hmdb": 5098, "ssv2": 5444}[name]
if limit: N = limit
enc = load_encoder()
CH = LO > 0 or HI < 10 ** 9
if CH and LO > 0:   # later chunk: reuse PCA and memmap written by the first chunk
    store = TokenStore.__new__(TokenStore)
    store.mm = np.load(f"{out}/tokens.npy", mmap_mode="r+")
    pz = np.load(f"{out}/pca.npz"); store.mu, store.W, store.buf, store.nfit = pz["mu"], pz["W"], [], 0
    store.evr = json.load(open(f"{out}/pca_evr.json"))["evr"]
else:
    store = TokenStore(f"{out}/tokens.npy", N, T)
Eres = np.zeros((N, T, NP), np.float32); Eraw = np.zeros((N, T, NP), np.float32)
labels = np.zeros(N, np.int64); split = np.zeros(N, np.int64); nraw = np.zeros(N, np.int64); metas = []
t0 = time.time()
for i, lab, s, fr, meta in items():
    if i >= N: break
    if not (LO <= i < HI): continue
    crops, grey = fr
    R = len(crops); idx = lattice(R, T)
    e, res = probe_energy(grey=grey)
    Eraw[i], Eres[i] = aggregate(e, idx), aggregate(res, idx)
    cand = np.stack([crops[j] for j in idx])
    store.put(i, encode(enc, cand), s == 0)
    labels[i], split[i], nraw[i] = lab, s, R
    meta.update(lattice=idx.tolist()); metas.append(meta)
    if i % 100 == 0:
        print(f"{name} {i}/{N} {time.time() - t0:.0f}s", flush=True)
store.close()
if CH:   # save this chunk; merge_chunks() assembles the final arrays
    np.savez(f"{out}/chunk_{LO}.npz", lo=LO, hi=min(HI, N), Eres=Eres[LO:HI], Eraw=Eraw[LO:HI], labels=labels[LO:HI],
             split=split[LO:HI], nraw=nraw[LO:HI], metas=json.dumps(metas))
    if LO == 0:
        np.savez(f"{out}/pca.npz", mu=store.mu, W=store.W); json.dump(dict(evr=store.evr), open(f"{out}/pca_evr.json", "w"))
    print("chunk done", LO, HI, time.time() - t0); sys.exit(0)
np.save(f"{out}/Eres.npy", Eres); np.save(f"{out}/Eraw.npy", Eraw)
np.save(f"{out}/labels.npy", labels); np.save(f"{out}/split.npy", split); np.save(f"{out}/nraw.npy", nraw)
np.savez(f"{out}/pca.npz", mu=store.mu, W=store.W)
json.dump(dict(T=T, N=N, pca_evr=store.evr, seconds=time.time() - t0, metas=metas), open(f"{out}/meta.json", "w"))
print("done", name, time.time() - t0, "evr", store.evr)
