"""Static tables (notation, related-work comparison, implementation)."""

NOTATION = ("tab", "notation", "Notation", [
    ["Symbol", "Meaning"],
    ["$R$, $I_r$", "number of raw frames; raw frame $r$"],
    ["$T$, $c_j$, $\\tau_j$", "lattice size; raw index and timestamp of candidate $j$"],
    ["$K$, $j_k$", "frame budget; lattice index of the $k$-th selected frame"],
    ["$N$, $D$", "patch regions per frame; token width"],
    ["$e_{r,n}$, $\\tilde{e}_{r,n}$", "cell energy; camera-compensated residual"],
    ["$E_{j,n}$, $v_j$", "lattice energy per region; lattice velocity"],
    ["$\\rho$, $\\gamma$", "coverage share; velocity exponent"],
    ["$p_j$, $F_j$", "sampling density; cumulative distribution"],
    ["$s_{k,n}$, $g_{\\phi}$", "token score; learned scorer"],
    ["$B$, $\\beta$, $\\mathcal{S}$", "patch-token budget; retained fraction; kept set"],
    ["$\\mathbf{c}_k$, $\\mu_k$", "context token of frame $k$; its mass"],
    ["$\\Delta_k$, $\\ell_k$", "sampling interval; log-interval code"],
    ["$\\mathbf{u}_{k,n}$", "feature velocity of a token"],
    ["$L$, $d$", "transformer depth; width"],
], [1500, 3500], {"upper": True})

RELATED_TAB = ("tab", "related", "Comparison of the closest frame and token selection methods with the proposed approach", [
    ["Method", "Selection signal", "Selected unit", "Decided before encoder", "Coverage guarantee", "Geometry given to model", "Limitation for long-form recognition"],
    ["MGSampler [@mgsampler]", "cumulative frame difference", "frames", "yes", "no", "no (index)", "camera motion inflates energy; quiet segments can be skipped"],
    ["Motion-guided sampler [@tmcsampler]", "robust motion cue", "frames", "yes", "no", "no", "all tokens of each frame kept"],
    ["OCSampler [@ocsampler]", "skim network, policy", "frames", "partly", "no", "no", "trained policy; preview over every frame"],
    ["Uni-AdaFocus [@uniadafocus]", "global encoder, policy", "frames and one crop", "partly", "no", "no", "fixed crop size; recurrent policy"],
    ["STTS [@stts]", "learned scorer", "tokens of uniform frames", "no", "no", "no", "cannot add frames inside a burst"],
    ["RLT [@rlt]", "repeated patches", "tokens", "yes", "n/a", "run length", "frames still sampled uniformly"],
    ["vid-TLDR [@vidtldr]", "attention sharpness", "tokens", "no", "n/a", "no", "needs early attention maps"],
    ["AKS, BOLT [@aks, bolt]", "query–frame similarity", "frames", "no", "relevance–coverage trade-off (AKS)", "no", "text query and features of every candidate"],
    ["V-CAST [@vcast]", "feature curvature", "per-frame token budget", "no", "no", "no", "serves a frozen video LLM"],
    ["KAIROS (this work)", "compensated pixel velocity, novelty, learned", "frames and tokens, one budget", "yes", "yes, Proposition 1", "time, interval, velocity", "probe blind to motion without intensity change"],
], [1550, 1450, 1350, 1000, 1150, 1150, 2668], {"upper": True, "size": 14})


def impl_table(evr):
    return ("tab", "impl", "Implementation settings", [
        ["Item", "Setting"],
        ["Encoder", "DINOv2 ViT-S/14, frozen, $112\\times112$ input"],
        ["Tokens", "1 class token, $8\\times8$ patches pooled to $N=16$; $D=192$ (PCA, " + evr + ")"],
        ["Lattice", "$T=48$ (TRACE-10), 32 (HMDB51), 24 (SSv2)"],
        ["Budgets", "$K=8$, $\\beta=0.25$, $B=24$, 41 input tokens"],
        ["Sampler", "$\\rho=0.2$, $\\gamma=1$, probe $56\\times56$ grey"],
        ["Temporal model", "$L=2$, $d=128$, 4 heads, MLP $2d$, dropout 0.1"],
        ["Scorer $g_{\\phi}$", "$192\\to64\\to1$, GELU"],
        ["Optimizer", "AdamW, lr $10^{-3}$, weight decay 0.05"],
        ["Schedule", "1-epoch warm-up, cosine, 20 epochs, batch 64"],
        ["Loss", "cross-entropy, label smoothing $\\epsilon=0.1$"],
        ["Selection", "best validation top-1; seeds 0, 1, 2"],
        ["Software", "Python 3.13, PyTorch 2.15, timm 1.0.31"],
        ["Hardware", "2-core CPU, one thread per run; Colab T4 for the notebook"],
    ], [1300, 3700], {"upper": True})
