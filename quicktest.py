"""Learnability check: encode only 8 oracle frames and 8 uniform frames per clip for a small set."""
import numpy as np, torch, sys
torch.set_num_threads(1)
from synth import render, R
from common import load_encoder, encode, crop_resize, NP
import kairos as kz
enc = load_encoder(); n = int(sys.argv[1])
lab = np.random.default_rng(5).permutation(np.arange(n) % 10)
Xo, Xu = [], []
for i in range(n):
    fr, m = render(10000 + i, int(lab[i]))
    o = np.round(np.linspace(m["t0"], m["t1"], 8)).astype(int); u = np.round(np.linspace(12, R - 13, 8)).astype(int)
    f = encode(enc, np.stack([fr[j] for j in np.concatenate([o, u])]))
    Xo.append(f[:8]); Xu.append(f[8:])
np.savez("quick.npz", Xo=np.array(Xo, np.float16), Xu=np.array(Xu, np.float16), y=lab)
