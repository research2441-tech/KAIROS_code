"""Samplers, token reducers, head network and training loop for velocity-proportional
spatiotemporal token allocation on frozen-encoder tokens."""
import math, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRID = 4          # head grid after 2x2 pooling of the 8x8 DINOv2 patch grid
NP = GRID * GRID


def pool_tokens(x):
    """(..., 65, D) -> (..., 1+NP, D): CLS kept, 8x8 patches average-pooled to GRIDxGRID."""
    if GRID == 8: return x
    c, p = x[..., :1, :], x[..., 1:, :]
    sh = p.shape[:-2]; f = 8 // GRID
    p = p.reshape(sh + (GRID, f, GRID, f, p.shape[-1])).mean((-4, -2)).reshape(sh + (NP, p.shape[-1]))
    return np.concatenate([c, p], -2)


def pool_motion(m):
    if GRID == 8: return m
    sh = m.shape[:-1]; f = 8 // GRID
    return m.reshape(sh + (GRID, f, GRID, f)).sum((-3, -1)).reshape(sh + (NP,))


# ----------------------------------------------------------------------------- data
def pool_file(root, chunk=200):
    """Writes tokens_p.npy: CLS + GRIDxGRID pooled patch tokens (float16), loaded fully into memory later."""
    src = np.load(f"{root}/tokens.npy", mmap_mode="r")
    out = np.lib.format.open_memmap(f"{root}/tokens_p.npy", mode="w+", dtype=np.float16, shape=src.shape[:2] + (1 + NP, src.shape[-1]))
    for i in range(0, len(src), chunk):
        out[i:i + chunk] = pool_tokens(np.asarray(src[i:i + chunk], np.float32)).astype(np.float16)
    out.flush()


class Bank:
    def __init__(self, root, light=False):
        import os
        if not os.path.exists(f"{root}/tokens_p.npy"):
            pool_file(root)
        self.tok = np.load(f"{root}/tokens_p.npy", mmap_mode="r" if light else None)
        self.Eres = pool_motion(np.load(f"{root}/Eres.npy")); self.Eraw = pool_motion(np.load(f"{root}/Eraw.npy"))
        self.Etm = np.maximum(self.Eres - np.median(self.Eres, 1, keepdims=True), 0)
        self.y = np.load(f"{root}/labels.npy"); self.split = np.load(f"{root}/split.npy")
        self.N, self.T = self.tok.shape[:2]; self.D = self.tok.shape[-1]
        self.C = int(self.y.max()) + 1
        self.root = root

    def ids(self, s):
        return np.where(self.split == s)[0]


# ----------------------------------------------------------------------------- samplers
def _strict(idx, T):
    idx = np.asarray(idx, np.int64).copy(); K = len(idx)
    for k in range(1, K):
        idx[k] = max(idx[k], idx[k - 1] + 1)
    for k in range(K - 1, -1, -1):
        idx[k] = min(idx[k], T - K + k)
        if k < K - 1: idx[k] = min(idx[k], idx[k + 1] - 1)
    return idx


def quantile_sample(v, K, rho, gamma, rng=None):
    """Velocity-quantile sampling: density (1-rho) v^gamma / sum + rho / T on the lattice."""
    T = len(v); w = np.power(np.maximum(v, 0) + 1e-6, gamma); w = w / w.sum()
    p = (1 - rho) * w + rho / T
    Fc = np.cumsum(p); Fc[-1] = 1.0
    u = rng.random(K) if rng is not None else np.full(K, 0.5)
    q = (np.arange(K) + u) / K
    return _strict(np.searchsorted(Fc, q, side="left"), T)


def uniform_sample(T, K, rng=None):
    edges = np.linspace(0, T, K + 1)
    if rng is None:
        idx = np.floor((edges[:-1] + edges[1:]) / 2)
    else:
        idx = np.floor(edges[:-1] + rng.random(K) * (edges[1:] - edges[:-1]))
    return _strict(np.clip(idx, 0, T - 1), T)


def tmed(E):
    """Remove the persistent (temporal-median) motion level of every region: (T, N) -> (T, N)."""
    return np.maximum(E - np.median(E, 0, keepdims=True), 0)


def select(bank, i, K, sampler, rng=None, Eres=None, Eraw=None, T=None):
    Eres = bank.Eres[i] if Eres is None else Eres
    Eraw = bank.Eraw[i] if Eraw is None else Eraw
    T = Eres.shape[0]
    if sampler["name"] == "uniform":
        return uniform_sample(T, K, rng)
    if sampler["name"] == "mg":      # cumulative raw motion, no compensation, no coverage floor
        return quantile_sample(Eraw.sum(1), K, 0.0, 1.0, rng)
    if sampler["name"] == "topk":    # K most active lattice points
        v = Eraw.sum(1) + (rng.random(T) * 1e-3 if rng is not None else 0)
        return np.sort(np.argsort(-v)[:K])
    if sampler["name"] == "vqs":
        E = tmed(Eres if sampler.get("comp", True) else Eraw)
        return quantile_sample(E.sum(1), K, sampler.get("rho", .2), sampler.get("gamma", 2.0), rng)
    raise ValueError(sampler)


def make_batch(bank, ids, K, sampler, rng=None, view=None):
    """Returns tokens (B,K,65,D), motion (B,K,64), tau (B,K), labels.
    `view(i)` may return a transformed clip (tokens, Eres, Eraw, tau_lattice) for stress protocols."""
    X, M, TAU = [], [], []
    for i in ids:
        if view is None:
            idx = select(bank, i, K, sampler, rng)
            X.append(bank.tok[i, idx].astype(np.float32)); M.append(bank.Etm[i, idx])
            TAU.append(idx / (bank.T - 1))
        else:
            tok, Er, Ew, tl = view(i)
            idx = select(bank, i, K, sampler, rng, Er, Ew)
            X.append(np.asarray(tok[idx], np.float32)); M.append(tmed(Er)[idx]); TAU.append(tl[idx])
    return (torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(M)).float(),
            torch.from_numpy(np.stack(TAU)).float(), torch.from_numpy(bank.y[ids]))


# ----------------------------------------------------------------------------- model
def fourier(x, nf=6):
    f = (2.0 ** torch.arange(nf, device=x.device, dtype=x.dtype)) * math.pi
    a = x[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1)


class Block(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))
        self.h, self.drop = h, drop

    def forward(self, x, bias):
        B, L, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=self.drop if self.training else 0.)
        x = x + self.o(a.transpose(1, 2).reshape(B, L, d))
        return x + self.mlp(self.n2(x))


class Head(nn.Module):
    """cfg keys: reducer in {full, cls, random, stts, tome, evit, rlt, ours}; budget B (patch tokens);
    enc in {index, ct} ; ctx (context tokens, ours) ; prior (motion+novelty prior, ours) ; glob (global budget)."""

    def __init__(self, D, C, cfg, d=128, L=2, h=4, drop=0.1, Kmax=64):
        super().__init__()
        self.cfg = cfg
        self.inp = nn.Linear(D, d)
        self.pos = nn.Parameter(torch.zeros(NP + 2, d)); nn.init.trunc_normal_(self.pos, std=.02)
        self.read = nn.Parameter(torch.zeros(1, 1, d))
        self.slot = nn.Embedding(Kmax, d)
        self.tenc, self.genc = nn.Linear(12, d), nn.Linear(12, d)
        self.vel = nn.Linear(D, d)
        self.g_geo, self.g_vel = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
        self.scorer = nn.Sequential(nn.Linear(D, 64), nn.GELU(), nn.Linear(64, 1))
        self.wm, self.wn = nn.Parameter(torch.tensor(1.0)), nn.Parameter(torch.tensor(1.0))
        self.blocks = nn.ModuleList([Block(d, h, drop) for _ in range(L)])
        self.norm, self.fc = nn.LayerNorm(d), nn.Linear(d, C)

    # ---- temporal encodings
    def temporal(self, tau, Kn):
        B, K = tau.shape
        enc = self.cfg.get("enc", "index")
        if enc == "index":
            return self.slot(torch.arange(K, device=tau.device))[None].expand(B, K, -1)
        gap = torch.cat([tau[:, :1] * 0 + 1.0 / max(Kn, 1), tau[:, 1:] - tau[:, :-1]], 1).clamp_min(1e-3)
        lg = torch.log(gap * Kn) / 4.0                       # 0 when spacing equals the uniform spacing
        if enc == "ctg":                                     # order embedding + zero-initialised gated geometry terms
            return (self.slot(torch.arange(K, device=tau.device))[None].expand(B, K, -1)
                    + self.g_geo * (self.tenc(fourier(tau)) + self.genc(fourier(lg))))
        if enc == "rel":                                     # order + intervals only (invariant to time translation)
            return self.slot(torch.arange(K, device=tau.device))[None].expand(B, K, -1) + self.genc(fourier(lg))
        return self.tenc(fourier(tau)) + self.genc(fourier(lg))

    def forward(self, X, M, tau):
        cfg = self.cfg; B, K, _, D = X.shape; dv = X.device
        cls_t, P = X[:, :, 0], X[:, :, 1:]                    # (B,K,D), (B,K,64,D)
        Prev = torch.cat([P[:, :1], P[:, :-1]], 1)
        te = self.temporal(tau, K)                            # (B,K,d)
        use_vel = cfg.get("enc", "index") in ("ct", "rel", "ctg") and cfg.get("vel", True)
        gv = self.g_vel if cfg.get("enc") == "ctg" else 1.0
        if use_vel:
            gap = torch.cat([torch.ones_like(tau[:, :1]), (tau[:, 1:] - tau[:, :-1]) * K], 1).clamp_min(.05)
            Vp = (P - Prev) / gap[..., None, None]
            Vc = (cls_t - torch.cat([cls_t[:, :1], cls_t[:, :-1]], 1)) / gap[..., None]
        red = cfg["reducer"]
        toks, bias = [self.read.expand(B, -1, -1)], [torch.zeros(B, 1, device=dv)]
        ec = self.inp(cls_t) + self.pos[0] + te + (gv * self.vel(Vc) if use_vel else 0)
        toks.append(ec); bias.append(torch.zeros(B, K, device=dv))
        if red == "cls":
            pass
        elif red == "full":
            e = self.inp(P) + self.pos[1:NP + 1] + te[:, :, None] + (gv * self.vel(Vp) if use_vel else 0)
            toks.append(e.reshape(B, K * NP, -1)); bias.append(torch.zeros(B, K * NP, device=dv))
        else:
            b = cfg["B"]; per = b // K
            if red == "tome":
                kept, mass, pid = self._tome(P, per)
                e = self.inp(kept) + self.pos[1:NP + 1][pid] + te[:, :, None]
                toks.append(e.reshape(B, -1, e.shape[-1])); bias.append(torch.log(mass).reshape(B, -1))
            else:
                s, extra = self._scores(P, Prev, M, cls_t, red)
                if red in ("ours",) and cfg.get("glob", True) or red == "rlt":
                    nb = b
                    flat = s.reshape(B, K * NP)
                    if red == "rlt":
                        flat = flat.clone(); flat[:, :NP] = 1e9   # first frame kept in full
                    top = flat.topk(nb, 1).indices
                    keep = torch.zeros(B, K * NP, dtype=torch.bool, device=dv).scatter_(1, top, True).view(B, K, NP)
                else:
                    top = s.topk(per, -1).indices
                    keep = torch.zeros(B, K, NP, dtype=torch.bool, device=dv).scatter_(2, top, True)
                kf = keep.reshape(B, -1); order = torch.sort(kf.float(), stable=True, dim=1, descending=True).indices[:, :int(kf.sum(1).min())]
                Pf = P.reshape(B, K * NP, D).gather(1, order[..., None].expand(-1, -1, D))
                kidx, nidx = order // NP, order % NP
                e = self.inp(Pf) + self.pos[1:NP + 1][nidx] + te.gather(1, kidx[..., None].expand(-1, -1, te.shape[-1]))
                if use_vel:
                    e = e + gv * self.vel(Vp.reshape(B, K * NP, D).gather(1, order[..., None].expand(-1, -1, D)))
                if red in ("stts", "ours"):              # straight-through: forward value 1, gradient to the scorer
                    g = torch.sigmoid(s.reshape(B, -1).gather(1, order))
                    e = e * (1 + g - g.detach())[..., None]
                toks.append(e); bias.append(torch.zeros(B, e.shape[1], device=dv))
                if red == "evit" or (red == "ours" and cfg.get("ctx", True)):
                    drop = (~keep).float()
                    cnt = drop.sum(-1)
                    ctx = (P * drop[..., None]).sum(2) / cnt.clamp_min(1)[..., None]
                    ec2 = self.inp(ctx) + self.pos[NP + 1] + te
                    toks.append(ec2); bias.append(torch.log(cnt.clamp_min(1e-4)))
        x = torch.cat(toks, 1)
        bb = torch.cat(bias, 1)[:, None, None, :].to(x.dtype)
        for blk in self.blocks:
            x = blk(x, bb)
        return self.fc(self.norm(x[:, 0]))

    def _scores(self, P, Prev, M, cls_t, red):
        B, K = P.shape[:2]
        if red == "random":
            return torch.rand(B, K, NP, device=P.device), None
        if red == "evit":
            return F.cosine_similarity(P, cls_t[:, :, None], dim=-1), None
        nov = 1 - F.cosine_similarity(P, Prev, dim=-1)       # first frame -> 0
        if red == "rlt":
            return nov, None
        learned = self.scorer(P).squeeze(-1)
        if red == "stts" or not self.cfg.get("prior", True):
            return learned, None

        def z(a):
            mu = a.mean((1, 2), keepdim=True); sd = a.std((1, 2), keepdim=True) + 1e-6
            return (a - mu) / sd
        return self.wm * z(torch.log1p(M)) + self.wn * z(nov) + learned, None

    @staticmethod
    def _tome(P, per):
        """Bipartite soft matching (training-free) per frame, applied until `per` tokens remain."""
        B, K, N, D = P.shape
        x = P.reshape(B * K, N, D); size = torch.ones(B * K, N, device=P.device); pid = torch.arange(N, device=P.device).expand(B * K, N).clone()
        while x.shape[1] > per:
            n = x.shape[1]; r = min(n // 2, n - per)
            a, bt = x[:, ::2], x[:, 1::2]; sa, sb = size[:, ::2], size[:, 1::2]
            sim = F.normalize(a, dim=-1) @ F.normalize(bt, dim=-1).transpose(1, 2)
            best, j = sim.max(-1)
            order = best.argsort(-1, descending=True)
            src, keep_a = order[:, :r], order[:, r:]
            dst = j.gather(1, src)
            ws = sa.gather(1, src)
            xb = bt * sb[..., None]
            xb = xb.scatter_add(1, dst[..., None].expand(-1, -1, D), a.gather(1, src[..., None].expand(-1, -1, D)) * ws[..., None])
            sb2 = sb.scatter_add(1, dst, ws)
            xb = xb / sb2[..., None]
            x = torch.cat([a.gather(1, keep_a[..., None].expand(-1, -1, D)), xb], 1)
            size = torch.cat([sa.gather(1, keep_a), sb2], 1)
            pid = torch.cat([pid[:, ::2].gather(1, keep_a), pid[:, 1::2]], 1)
        return x.view(B, K, per, D), size.view(B, K, per), pid.view(B, K, per)


class MeanCLS(nn.Module):
    """TSN-style consensus: average of frame CLS features followed by a linear classifier."""

    def __init__(self, D, C, cfg=None):
        super().__init__(); self.n = nn.LayerNorm(D); self.fc = nn.Linear(D, C); self.cfg = cfg or {}

    def forward(self, X, M, tau):
        return self.fc(self.n(X[:, :, 0].mean(1)))


# ----------------------------------------------------------------------------- train / eval
def build(bank, cfg):
    if cfg["reducer"] == "mean":
        return MeanCLS(bank.D, bank.C, cfg)
    return Head(bank.D, bank.C, cfg)


@torch.no_grad()
def predict(model, bank, ids, K, sampler, bs=128, view=None):
    model.eval(); out = []
    for j in range(0, len(ids), bs):
        X, M, tau, _ = make_batch(bank, ids[j:j + bs], K, sampler, None, view)
        out.append(F.softmax(model(X.to(DEV), M.to(DEV), tau.to(DEV)), -1).cpu())
    return torch.cat(out).numpy()


def train(bank, cfg, seed=0, epochs=20, K=8, bs=64, lr=1e-3, log=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = build(bank, cfg).to(DEV)
    sampler = cfg["sampler"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    tr, va = bank.ids(0), bank.ids(1)
    steps = epochs * math.ceil(len(tr) / bs); warm = math.ceil(len(tr) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    best, best_state, t0 = -1, None, time.time()
    for ep in range(epochs):
        model.train(); perm = rng.permutation(tr)
        for j in range(0, len(perm), bs):
            X, M, tau, y = make_batch(bank, perm[j:j + bs], K, sampler, rng)
            loss = F.cross_entropy(model(X.to(DEV), M.to(DEV), tau.to(DEV)), y.to(DEV), label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        acc = (predict(model, bank, va, K, sampler).argmax(1) == bank.y[va]).mean()
        if acc > best:
            best, best_state = acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if log: log(f"ep{ep} val {acc:.4f} {time.time() - t0:.0f}s")
    model.load_state_dict(best_state)
    return model, best, time.time() - t0
