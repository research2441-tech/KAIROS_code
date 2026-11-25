"""Job runner. usage: python run_exp.py <plan> <worker_id> <n_workers>
Each job trains one configuration on one dataset/seed and stores validation accuracy,
test probabilities and stress-protocol probabilities under results/."""
import sys, os, json, time, numpy as np, torch
torch.set_num_threads(1)
import kairos as kz
from plans import PLANS, METHODS

plan, wid, nw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
plan, _, only = plan.partition(":")
jobs = [j for j in PLANS[plan]() if not only or j["ds"] == only]
BANKS = {}


def bank(ds):
    if ds not in BANKS:
        BANKS[ds] = kz.Bank(f"data/{ds}")
    return BANKS[ds]


def dilution_view(b, phi, seed=0):
    """Idle-stream protocol: the clip occupies a fraction phi of the lattice and the remaining
    slots repeat its first/last frame with sensor-level probe noise."""
    T = b.T
    def view(i):
        rng = np.random.default_rng(seed * 7919 + int(i))
        n = max(2, int(round(T * phi)))
        src = np.round(np.linspace(0, T - 1, n)).astype(int)
        off = int(rng.integers(0, T - n + 1))
        slot = np.empty(T, int); slot[:off] = src[0]; slot[off:off + n] = src; slot[off + n:] = src[-1]
        tok = b.tok[i][slot]
        Er = np.zeros((T, b.Eres.shape[2]), np.float32); Ew = np.zeros_like(Er)
        cr, cw = np.cumsum(b.Eres[i], 0), np.cumsum(b.Eraw[i], 0)
        for j in range(n):
            lo = src[j - 1] if j > 0 else src[0] - 1
            Er[off + j] = cr[src[j]] - (cr[lo] if lo >= 0 else 0)
            Ew[off + j] = cw[src[j]] - (cw[lo] if lo >= 0 else 0)
        s0 = np.percentile(b.Eraw[i], 5) + 1e-3
        mask = np.ones(T, bool); mask[off:off + n] = False
        Ew[mask] = np.abs(rng.normal(0, s0, (mask.sum(), Ew.shape[1])))
        Er[mask] = np.maximum(Ew[mask] - np.median(Ew[mask], 1, keepdims=True), 0)
        return tok, Er, Ew, np.arange(T) / (T - 1)
    return view


def noise_view(b, sigma, seed=0):
    def view(i):
        rng = np.random.default_rng(seed * 104729 + int(i))
        s = sigma * b.Eraw[i].mean()
        Ew = np.abs(b.Eraw[i] + rng.normal(0, s, b.Eraw[i].shape)).astype(np.float32)
        Er = np.maximum(b.Eres[i] + rng.normal(0, s, b.Eres[i].shape), 0).astype(np.float32)
        return b.tok[i], Er, Ew, np.arange(b.T) / (b.T - 1)
    return view


def interp_slots(model, K_new, K_train):
    """Linear interpolation of learned slot embeddings for index-encoded heads evaluated at another K."""
    if not hasattr(model, "slot") or model.cfg.get("enc", "index") != "index" or K_new == K_train:
        return None
    W = model.slot.weight.data[:K_train].clone()
    pos = torch.linspace(0, K_train - 1, K_new)
    lo = pos.floor().long().clamp(max=K_train - 1); hi = (lo + 1).clamp(max=K_train - 1); a = (pos - lo)[:, None]
    old = model.slot.weight.data.clone()
    model.slot.weight.data[:K_new] = (1 - a) * W[lo] + a * W[hi]
    return old


for n, job in enumerate(jobs):
    if n % nw != wid:
        continue
    out = f"results/{job['ds']}/{job['name']}/seed{job['seed']}"
    if os.path.exists(out + "/done.json"):
        continue
    os.makedirs(out, exist_ok=True)
    try:                                  # lock so that several workers can share one plan
        fd = os.open(out + "/lock", os.O_CREAT | os.O_EXCL); os.close(fd)
    except FileExistsError:
        continue
    b = bank(job["ds"]); cfg = dict(job["cfg"])
    K = cfg.pop("K", 8)
    if K == "T": K = b.T
    logf = open(out + "/log.txt", "w")
    model, val, secs = kz.train(b, cfg, seed=job["seed"], epochs=job.get("epochs", 20), K=K,
                                log=lambda s: (logf.write(s + "\n"), logf.flush()))
    te = b.ids(2)
    t0 = time.time(); P = kz.predict(model, b, te, K, cfg["sampler"]); inf_s = time.time() - t0
    np.save(out + "/test_probs.npy", P.astype(np.float16))
    acc = float((P.argmax(1) == b.y[te]).mean())
    res = dict(job=job, K=K, val=float(val), test=acc, train_s=secs, infer_s=inf_s,
               params=int(sum(p.numel() for p in model.parameters())))
    for ex in job.get("extra", []):
        if ex[0] == "dilution":
            P2 = kz.predict(model, b, te, K, cfg["sampler"], view=dilution_view(b, ex[1]))
            np.save(out + f"/dil{ex[1]}.npy", P2.astype(np.float16)); res[f"dil{ex[1]}"] = float((P2.argmax(1) == b.y[te]).mean())
        elif ex[0] == "noise":
            P2 = kz.predict(model, b, te, K, cfg["sampler"], view=noise_view(b, ex[1]))
            res[f"noise{ex[1]}"] = float((P2.argmax(1) == b.y[te]).mean())
        elif ex[0] == "K":
            old = interp_slots(model, ex[1], K)
            P2 = kz.predict(model, b, te, ex[1], cfg["sampler"])
            if old is not None: model.slot.weight.data.copy_(old)
            res[f"K{ex[1]}"] = float((P2.argmax(1) == b.y[te]).mean())
    if job.get("save_model"):
        torch.save(model.state_dict(), out + "/model.pt")
    json.dump(res, open(out + "/done.json", "w"), indent=1)
    print(job["ds"], job["name"], job["seed"], f"val {val:.4f} test {acc:.4f} {secs:.0f}s", flush=True)
