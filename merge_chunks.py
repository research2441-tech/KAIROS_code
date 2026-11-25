import sys, glob, json, numpy as np
out = f"data/{sys.argv[1]}"; T = int(sys.argv[2])
ch = sorted(glob.glob(f"{out}/chunk_*.npz"), key=lambda f: int(np.load(f)["lo"]))
A = {k: np.concatenate([np.load(f)[k] for f in ch]) for k in ["Eres", "Eraw", "labels", "split", "nraw"]}
for k, v in A.items(): np.save(f"{out}/{k}.npy", v)
metas = sum([json.loads(str(np.load(f)["metas"])) for f in ch], [])
json.dump(dict(T=T, N=len(A["labels"]), pca_evr=json.load(open(f"{out}/pca_evr.json"))["evr"], metas=metas), open(f"{out}/meta.json", "w"))
print("merged", len(A["labels"]))
