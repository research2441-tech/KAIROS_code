"""Measured FLOPs (torch FlopCounterMode, multiply-add counted as 2 FLOPs) and CPU latency."""
import json, time, numpy as np, torch
from torch.utils.flop_counter import FlopCounterMode
import kairos as kz
from plans import METHODS
from common import load_encoder, probe_energy

torch.set_num_threads(1)
out = {}
enc = load_encoder()
x = torch.randn(1, 3, 112, 112)
with FlopCounterMode(display=False) as fc:
    enc.forward_features(x)
out["enc_gflops_per_frame"] = fc.get_total_flops() / 1e9
with torch.inference_mode():
    xb = torch.randn(8, 3, 112, 112); enc.forward_features(xb)
    t = time.time()
    for _ in range(5): enc.forward_features(xb)
    out["enc_ms_per_frame"] = (time.time() - t) / 40 * 1e3
# probe: measured time on 240x320 frames, analytic op count
fr = [np.random.randint(0, 255, (240, 320, 3), np.uint8) for _ in range(64)]
t = time.time(); probe_energy(fr); out["probe_ms_per_frame"] = (time.time() - t) / 64 * 1e3
out["probe_mflops_per_frame"] = (240 * 320 * 3 * 2 + 112 * 112 * 5 + 112 * 112 * 2 + 56 * 56 * 3) / 1e6
# heads
D, C, K = 192, 51, 8
heads = {}
for name in ["Uniform-CLS", "Dense-CLS", "Uniform-Full", "Uniform-STTS", "Uniform-ToMe", "Uniform-EViT", "Uniform-RLT", "Ours",
             "Sens-K4", "Sens-K16", "Sens-U-Full-K4", "Sens-U-Full-K16", "Sens-beta0.125", "Sens-beta0.5", "Sens-beta1.0"]:
    cfg = dict(METHODS[name]); k = cfg.pop("K"); k = 32 if k == "T" else k
    m = kz.build(type("B", (), dict(D=D, C=C))(), cfg).eval()
    X = torch.randn(1, k, 1 + kz.NP, D); M = torch.rand(1, k, kz.NP); tau = torch.linspace(0, 1, k)[None]
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            m(X, M, tau)
        m(X, M, tau); t = time.time()
        for _ in range(20): m(X, M, tau)
    heads[name] = dict(K=k, gflops=fc.get_total_flops() / 1e9, ms=(time.time() - t) / 20 * 1e3,
                       params=sum(p.numel() for p in m.parameters()))
out["heads"] = heads
json.dump(out, open("results/flops.json", "w"), indent=1)
print(json.dumps(out, indent=1))
