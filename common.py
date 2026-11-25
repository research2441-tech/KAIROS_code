"""Shared utilities: frozen DINOv2-S/14 encoder, cheap pixel motion probe, PCA token store."""
import numpy as np, torch, timm, cv2

IMG = 112          # encoder input resolution (8x8 patch grid with patch size 14)
GRID = 8           # patch grid side
NP = GRID * GRID   # patches per frame
PROBE = 56         # probe resolution (7x7 pixel cells)
DPCA = 192         # stored token dimension after PCA
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder():
    m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                          img_size=IMG, num_classes=0).eval().to(DEV)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def crop_resize(fr, size=IMG):
    """Resize short side to `size`, then centre crop (uint8 RGB in, uint8 out)."""
    h, w = fr.shape[:2]
    s = size / min(h, w)
    nh, nw = max(size, round(h * s)), max(size, round(w * s))
    r = cv2.resize(fr, (nw, nh), interpolation=cv2.INTER_AREA)
    y0, x0 = (nh - size) // 2, (nw - size) // 2
    return r[y0:y0 + size, x0:x0 + size]


def probe_grey(crop112):
    return cv2.resize(cv2.cvtColor(crop112, cv2.COLOR_RGB2GRAY), (PROBE, PROBE), interpolation=cv2.INTER_AREA)


def probe_energy(frames_rgb=None, grey=None):
    """Per-raw-frame, per-cell absolute-difference energy on a 56x56 grey probe.
    Returns e_raw (R, 64) and the camera-compensated residual e_res (R, 64)."""
    if grey is None:
        grey = [probe_grey(crop_resize(f, IMG)) for f in frames_rgb]
    g = np.stack(grey).astype(np.float32)
    d = np.zeros_like(g)
    d[1:] = np.abs(g[1:] - g[:-1])
    c = PROBE // GRID
    e = d.reshape(len(g), GRID, c, GRID, c).mean(axis=(2, 4)).reshape(len(g), NP)
    res = np.maximum(e - np.median(e, axis=1, keepdims=True), 0.0)
    return e, res


def lattice(R, T):
    return np.round(np.linspace(0, R - 1, T)).astype(np.int64)


def aggregate(e, idx):
    """Sum raw-frame energy over the interval (idx[j-1], idx[j]] of each lattice point."""
    cs = np.concatenate([np.zeros((1, e.shape[1]), e.dtype), np.cumsum(e, 0)])
    out = np.zeros((len(idx), e.shape[1]), np.float32)
    prev = idx[0]
    for j, i in enumerate(idx):
        lo = prev + 1 if j > 0 else i
        out[j] = cs[i + 1] - cs[lo] if i >= lo else 0.0
        prev = i
    return out


@torch.inference_mode()
def encode(model, frames112):
    x = (frames112.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
    out = []
    for i in range(0, len(x), 64):
        # bfloat16 autocast (AMX on recent Xeon CPUs, tensor cores on GPU); features stay within ~2% of fp32
        with torch.autocast(DEV, dtype=torch.bfloat16):
            out.append(model.forward_features(x[i:i + 64].to(DEV)).float().cpu())  # (b, 65, 384) CLS + 64 patches
    return torch.cat(out).numpy()


class TokenStore:
    """Writes (N, T, 65, DPCA) float16 tokens; PCA is fitted on the first `nfit` training clips."""

    def __init__(self, path, N, T, nfit=120):
        self.mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(N, T, 1 + NP, DPCA))
        self.buf, self.nfit, self.mu, self.W = [], nfit, None, None

    def _proj(self, f):
        return ((f.reshape(-1, f.shape[-1]) - self.mu) @ self.W).reshape(f.shape[:-1] + (DPCA,)).astype(np.float16)

    def put(self, i, f, is_train):
        if self.W is None:
            self.buf.append((i, f.astype(np.float16), is_train))
            if sum(b[2] for b in self.buf) >= self.nfit:
                self.fit()
            return
        self.mm[i] = self._proj(f)

    def fit(self):
        X = np.concatenate([b[1].reshape(-1, b[1].shape[-1]) for b in self.buf if b[2]]).astype(np.float32)
        rng = np.random.default_rng(0)
        X = X[rng.choice(len(X), min(len(X), 120000), replace=False)]
        self.mu = X.mean(0)
        U, S, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.W = Vt[:DPCA].T.astype(np.float32)
        self.evr = float((S[:DPCA] ** 2).sum() / (S ** 2).sum())
        for i, f, _ in self.buf:
            self.mm[i] = self._proj(f.astype(np.float32))
        self.buf = []

    def close(self):
        if self.W is None:
            self.fit()
        self.mm.flush()
