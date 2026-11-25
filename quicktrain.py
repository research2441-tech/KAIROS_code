import numpy as np, torch, torch.nn.functional as F, sys
torch.set_num_threads(1)
import kairos as kz
d = np.load("quick.npz"); y = torch.from_numpy(d["y"]).long(); n = len(y); tr = np.arange(int(n*.7)); te = np.arange(int(n*.7), n)
for key in ["Xo", "Xu"]:
    X = torch.from_numpy(kz.pool_tokens(d[key].astype(np.float32)))
    for red in ["cls", "full"]:
        torch.manual_seed(0)
        m = kz.Head(384, 10, dict(reducer=red, enc="ct")); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=0.05)
        M = torch.zeros(n, 8, kz.NP); tau = torch.linspace(0, 1, 8).expand(n, 8)
        for ep in range(40):
            m.train(); p = np.random.permutation(tr)
            for j in range(0, len(p), 64):
                b = p[j:j+64]; loss = F.cross_entropy(m(X[b], M[b], tau[b]), y[b]); opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): acc = (m(X[te], M[te], tau[te]).argmax(1) == y[te]).float().mean().item()
        print(key, red, f"{acc:.3f}", flush=True)
