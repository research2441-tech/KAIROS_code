"""Results, discussion, limitations, conclusion, front matter. Every number is read from results/summary.json."""
import json, numpy as np

S = json.load(open("results/summary.json")); FL = json.load(open("results/flops.json"))
R = S["runs"]
P = lambda ds, n, k="test": 100 * R[ds][n][k][0]
SD = lambda ds, n, k="test": 100 * R[ds][n][k][1]
f1 = lambda x: f"{x:.1f}"
DS = ["trace", "hmdb", "ssv2"]
MAINB = ["TSN-mean", "Uniform-CLS", "Dense-CLS", "Uniform-Full", "MG-Full", "Uniform-Random", "Uniform-STTS", "Uniform-ToMe",
         "Uniform-EViT", "Uniform-RLT", "MG-STTS"]
NICE = {"TSN-mean": "the TSN-style mean", "Uniform-CLS": "Uniform-CLS", "Dense-CLS": "Dense-CLS", "Uniform-Full": "Uniform-Full",
        "MG-Full": "MG-Full", "Uniform-Random": "random dropping", "Uniform-STTS": "the STTS-style selector", "Uniform-ToMe": "ToMe",
        "Uniform-EViT": "EViT", "Uniform-RLT": "the RLT-style reducer", "MG-STTS": "MG + STTS", "Ours": "KAIROS"}


W = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 11: "all 11"}


def best_base(ds, pool=MAINB):
    return max(pool, key=lambda n: R[ds][n]["test"][0])


def gfl(ds, n):
    from figs_results import gflops
    return gflops(ds, n)


def st(ds, n):
    return S["stats"][ds][n]


def pv(p):
    return "p<0.001" if p < 1e-3 else f"p={p:.3f}"


CAP = {
 "arch": "Overall pipeline of KAIROS. The upper row works on pixels before any deep encoding: a $56\\times56$ probe with a spatial median (camera motion) and a temporal median (persistent motion) produces a velocity field over a candidate lattice, and the coverage-floored sampler chooses $K$ lattice frames; only those frames are decoded at full resolution and passed to the frozen DINOv2 ViT-S/14. The lower row is trained: patch tokens compete for one clip-level budget $B$, dropped patches survive as mass-weighted context tokens, and each token carries an order embedding plus gated encodings of timestamp, sampling interval and feature velocity before a two-layer transformer. Solid arrows show forward data flow; dashed arrows show the reverse-mode gradient, which reaches the scorer through the straight-through gate and stops at the non-differentiable sampler and at the frozen encoder.",
 "sampler": "The sampler on one video from the TRACE-10 generator, outside the benchmark splits (fast right jab at frames 45–55, camera bump at frame 97). (a) Raw difference energy and the camera-compensated residual; the bump dominates the raw signal. (b) Cumulative distributions over the $T=48$ lattice: raw cumulative motion (dashed) and the coverage-floored density of {eq:dens} after the temporal median (solid); horizontal lines are the stratified quantile targets. (c) Frames chosen by each sampler with the number that fall inside the event.",
 "tokens": "Retained regions for one TRACE-10 test video under the STTS-style selector on uniformly sampled frames (top) and under KAIROS (bottom). Each panel shows a selected frame in grayscale with its raw index; retained $28\\times28$ regions are outlined and dropped regions are lightened. KAIROS places several frames inside the event and spends most of the clip-level budget on them.",
 "pareto": "Top-1 accuracy of every method on the three datasets (mean over three seeds; bars show one standard deviation). The dotted line marks the best baseline. All methods except Dense-CLS use eight encoder calls; Dense-CLS encodes every lattice frame (four to six times as many). Costs are listed in Table {tab:main}.",
 "ablation": "Factorial ablation of the sampler (A), the cross-frame budget (B) and the sampling-geometry encoding (C), reported as the change in top-1 accuracy with respect to the configuration without any of them (mean and standard deviation over three seeds).",
 "sensitivity": "Sensitivity of KAIROS (seed 0) to the coverage share $\\rho$, the velocity exponent $\\gamma$, the retained-token fraction $\\beta$ and the frame budget $K$. Grey curves show the STTS-style selector at the same $\\beta$ and Uniform-Full at the same $K$.",
 "robust": "Robustness. (a)–(b) Idle-stream dilution: each test clip is compressed into a fraction $\\varphi$ of the lattice and the rest repeats its first or last frame with sensor-level probe noise. (c) Frame-count transfer: models trained with $K=8$ are evaluated with 4 to 16 frames at a fixed token budget; index-encoded models use linearly interpolated slot embeddings.",
 "cd": "Critical-difference diagram of mean ranks across the nine dataset–seed blocks (Nemenyi test, $\\alpha=0.05$). Methods joined by the thick bar do not differ significantly from the best-ranked method.",
 "hitdur": "Burst hit rate on the TRACE-10 test videos, split into fast events (10–16 frames) and slow events (64–96 frames). (a) Share of the eight selected frames that fall inside the event. (b) Share of videos with at least two selected frames inside the event.",
 "scaling": "Analytic cost of {eq:cost} as the number of frames grows, for a 12-layer temporal transformer of width 768 on a $4\\times4$ region grid: frozen encoder passes, joint attention over all tokens, and the clip-level budget with $\\beta=1/4$ (logarithmic axes; FLOPs counted as two per multiply–accumulate).",
 "perclass": "(a) Per-class difference between KAIROS and Uniform-Full on HMDB51, sorted. (b) Per-class accuracy on TRACE-10 for Uniform-Full, MG + STTS and KAIROS (mean over seeds).",
}

FRONT = dict(
 title="TITLE",
 authors="[First Author], [Second Author], [Third Author], and [Corresponding Author]",
 affil="[Department, Institution, City, Country]  |  Corresponding author: [email]  |  ORCID: [0000-0000-0000-0000]",
 abstract="ABSTRACT",
 keywords="KEYWORDS",
)
RESULTS = lambda TR: []
DISCUSSION, LIMITS, CONCLUSION, BACKMATTER = [], [], [], []


def rank_of(ds, n="Ours", pool=None):
    pool = pool or (MAINB + ["Ours"])
    order = sorted(pool, key=lambda m: -R[ds][m]["test"][0])
    return order.index(n) + 1, len(pool)


def phrase_vs(ds, n):
    """'x.x points above/below <name> (95% CI [a, b], p_Holm=...)'"""
    s = st(ds, n); d = 100 * s["delta"]
    side = "above" if d > 0 else "below"
    return f"{abs(d):.1f} points {side} {NICE[n]} (95% CI [{100 * s['lo']:+.1f}, {100 * s['hi']:+.1f}], Holm-adjusted {pv(s['p_holm'])})"


def V():
    """Commonly used values."""
    v = {}
    enc = FL["enc_gflops_per_frame"]; v["enc8"] = 8 * enc
    for ds in DS:
        v[f"bb_{ds}"] = best_base(ds)
        v[f"rk_{ds}"] = rank_of(ds)
    return v


EX = json.load(open("results/extra.json"))


def RESULTS(TR):
    v = V(); B = []
    fi = json.load(open("results/fig3_info.json")); t0, t1 = fi["event"]
    ins = lambda fr: [t0 <= r < t1 + 4 for r in fr]
    F3 = dict(ev=f"event at raw frames {t0}–{t1}", u_in=sum(ins(fi["Uniform-STTS"]["frames"])),
              k_in=sum(ins(fi["Ours"]["frames"])), k_tok=sum(k for k, i in zip(fi["Ours"]["kept_per_frame"], ins(fi["Ours"]["frames"])) if i))
    bt, bh, bs = v["bb_trace"], v["bb_hmdb"], v["bb_ssv2"]
    enc = FL["enc_gflops_per_frame"]
    B += [("h1", "V. RESULTS"),
          ("p", "The results are organized by the questions that the method raises: whether the allocation improves recognition at a matched budget (Section V-A), whether the sampler actually finds short events (V-B), which component is responsible (V-C), how sensitive the method is to its settings (V-D), how it behaves under stress (V-E), what it costs (V-F), how well it is calibrated (V-G), and whether the differences survive statistical testing (V-H). All numbers are test-split results averaged over three seeds unless stated otherwise."),
          ("h2", "A. Comparison at a Matched Budget"),
          ("tab_main",),
          ("p", f"Table {{tab:main}} and Fig. {{fig:pareto}} summarize the comparison. On TRACE-10, KAIROS reaches {f1(P('trace','Ours'))} ± {f1(SD('trace','Ours'))}% top-1 accuracy. The strongest baseline, {NICE[bt]}, reaches {f1(P('trace',bt))}%, so the margin is {f1(P('trace','Ours')-P('trace',bt))} points (bootstrap 95% CI [{100*st('trace',bt)['lo']:+.1f}, {100*st('trace',bt)['hi']:+.1f}]). With uniform sampling, the methods that select or merge tokens by appearance stay between {f1(min(P('trace',n) for n in ['Uniform-CLS','Uniform-Full','Uniform-Random','Uniform-STTS','Uniform-ToMe','Uniform-EViT']))}% and {f1(max(P('trace',n) for n in ['Uniform-CLS','Uniform-Full','Uniform-Random','Uniform-STTS','Uniform-ToMe','Uniform-EViT']))}%, close to the 10% chance level; the RLT-style reducer, which keeps the patches that change most between frames, reaches {f1(P('trace','Uniform-RLT'))}% because it retains whatever part of an event happens to fall on a sampled frame; full space-time attention over all 137 tokens (Uniform-Full, {f1(P('trace','Uniform-Full'))}%) does not help because the evidence is absent from the input. Encoding all 48 lattice frames (Dense-CLS, {f1(P('trace','Dense-CLS'))}%) costs six times the encoder calls and still trails KAIROS by {f1(P('trace','Ours')-P('trace','Dense-CLS'))} points, because the class token of a frame summarizes the whole image and dilutes a small moving object. Motion-guided sampling alone (MG-Full, {f1(P('trace','MG-Full'))}%) and its naive combination with learned token selection (MG + STTS, {f1(P('trace','MG-STTS'))} ± {f1(SD('trace','MG-STTS'))}%) recover part of the gap, which shows that frame choice matters, but they remain far from the full method."),
          ("p", f"The trimmed benchmarks behave differently. On HMDB51 all twelve methods lie within {f1(max(P('hmdb',n) for n in MAINB+['Ours'])-min(P('hmdb',n) for n in MAINB+['Ours']))} points of each other. The best result, {f1(P('hmdb',bh))}%, comes from {NICE[bh]}, and KAIROS ranks {v['rk_hmdb'][0]} of {v['rk_hmdb'][1]} with {f1(P('hmdb','Ours'))} ± {f1(SD('hmdb','Ours'))}%, {phrase_vs('hmdb', bh)}. The TSN-style mean of frame class tokens, which has no temporal model at all, is within {f1(abs(P('hmdb','TSN-mean')-P('hmdb',bh)))} points of the best method, which confirms that labels in this dataset are largely carried by appearance and that a frozen image encoder already captures most of it. On the SSv2 subset, KAIROS obtains {f1(P('ssv2','Ours'))} ± {f1(SD('ssv2','Ours'))}% top-1 and ranks {v['rk_ssv2'][0]} of {v['rk_ssv2'][1]}; the best baseline is {NICE[bs]} with {f1(P('ssv2',bs))}%, and KAIROS is {phrase_vs('ssv2', bs)}. Absolute SSv2 accuracies are low for every method, which is expected with about 22 training clips per class and a frozen encoder."),
          ("p", f"In cost, every method except Dense-CLS performs eight encoder calls ({v['enc8']:.1f} GFLOPs), so the totals in Table {{tab:main}} differ mostly through the temporal model and the probe. KAIROS feeds 41 tokens to the temporal model instead of 137 for full attention, which reduces the temporal-model cost from {FL['heads']['Uniform-Full']['gflops']:.3f} to {FL['heads']['Ours']['gflops']:.3f} GFLOPs, while the probe adds about {S['data']['hmdb']['nraw_mean']*FL['probe_mflops_per_frame']/1e3:.3f} GFLOPs on HMDB51. Dense-CLS needs {32*enc:.1f} GFLOPs on HMDB51 and {48*enc:.1f} GFLOPs on TRACE-10. The temporal-model saving is therefore real but small at this clip length; it grows quadratically with the number of frames, as discussed in Section VI."),
          ("h2", "B. Does the Sampler Find the Events?"),
          ("tab_hit",),
          ("p", f"Table {{tab:hit}} answers this question directly on the 600 TRACE-10 test videos, using {{eq:hit}}. Uniform sampling puts {100*S['hit']['Uniform']['H']:.1f}% of its eight frames inside the event and places two or more frames there in only {100*S['hit']['Uniform']['two']:.1f}% of the videos, too few to see the direction of a jab or a tilt. MGSampler raises the hit rate to {S['hit']['MGSampler']['H']:.3f}, and the coverage-floored sampler of KAIROS to {S['hit']['VQS (KAIROS)']['H']:.3f}, with at least two frames inside the event in {100*S['hit']['VQS (KAIROS)']['two']:.1f}% of the videos. Removing the camera compensation lowers the hit rate on videos with a camera bump from {S['hit']['VQS (KAIROS)']['H_shake']:.3f} to {S['hit']['VQS, no compensation']['H_shake']:.3f} and raises the share of frames spent on the bump from {S['hit']['VQS (KAIROS)']['shake']:.3f} to {S['hit']['VQS, no compensation']['shake']:.3f}. Without the coverage floor ($\\rho=0$) the hit rate is higher still ({S['hit']['VQS, rho=0']['H']:.3f}), because the floor deliberately keeps part of the budget elsewhere; Section V-C shows what this trade costs in accuracy. Fig. {{fig:tokens}} illustrates one test video ({F3['ev']}): uniform sampling catches the event in {W[F3['u_in']]} frame{'s' if F3['u_in'] != 1 else ''}, and the STTS-style selector keeps four regions in every frame, whereas KAIROS places {W[F3['k_in']]} frames inside the event and spends {F3['k_tok']} of its 24 patch tokens on them; only KAIROS predicts the correct class."),
          ("fig_tokens",),
          ("fig_hitdur",),
          ("p", f"Fig. {{fig:hitdur}} separates fast and slow events. For the 10–16-frame events, uniform sampling never places two frames inside the event and MGSampler does so in {100*EX['MGSampler']['fast']['two']:.0f}% of the videos, against {100*EX['KAIROS']['fast']['two']:.0f}% for KAIROS. For the 64–96-frame slow events all samplers place at least two frames inside them, so the slow classes test the encoding of elapsed time rather than the sampler."),
          ]
    return B


def ab(ds, a, b, c):
    return P(ds, "Ours" if (a, b, c) == (1, 1, 1) else f"Abl-A{a}B{b}C{c}")


def cfgname(t):
    a, b, c = t
    parts = [p for p, on in (("the sampler", a), ("the budget", b), ("the encoding", c)) if on]
    return ("the combination of " + " and ".join(parts) + (" with uniform sampling" if not a else "")) if parts else "the base configuration"


def RESULTS2():
    B = []
    t = lambda a, b, c: ab("trace", a, b, c)
    B += [("h2", "C. Ablation Study"),
          ("tab_abl",), ("fig_ablation",),
          ("p", f"The factorial ablation in Table {{tab:abl}} and Fig. {{fig:ablation}} switches each of the three components on and off. Component A is the sampler, B the cross-frame budget with the motion-novelty prior, and C the gated sampling-geometry encoding. On TRACE-10 no single component is enough. The configuration without any of them reaches {f1(t(0,0,0))}%, and adding only the sampler, only the budget or only the encoding gives {f1(t(1,0,0))}%, {f1(t(0,1,0))}% and {f1(t(0,0,1))}%. The components interact strongly: the sampler with the encoding reaches {f1(t(1,0,1))}%, the sampler with the budget {f1(t(1,1,0))}%, and all three together {f1(t(1,1,1))}%. The interaction has a simple reading. The sampler brings frames from inside the event into the input, the budget makes sure that the event frames keep their object tokens instead of sharing the budget evenly with near-duplicate frames, and the encoding tells the network that consecutive event frames are close in time, which is what separates fast from slow jabs. Slow-jab accuracy illustrates the last point: it rises from {f1(100*R['trace']['Abl-A1B1C0']['acc_slow'])}% without the encoding to {f1(100*R['trace']['Ours']['acc_slow'])}% with it."),
          ("p", f"On the trimmed benchmarks the same factorial shows much smaller effects. On HMDB51 the eight configurations span {f1(max(ab('hmdb',a,b,c) for a in (0,1) for b in (0,1) for c in (0,1))-min(ab('hmdb',a,b,c) for a in (0,1) for b in (0,1) for c in (0,1)))} points and on SSv2 {f1(max(ab('ssv2',a,b,c) for a in (0,1) for b in (0,1) for c in (0,1))-min(ab('ssv2',a,b,c) for a in (0,1) for b in (0,1) for c in (0,1)))} points, comparable to the seed standard deviations. The full configuration is not the best one on either dataset: on both, the highest accuracy comes from {cfgname(max(((a,b,c) for a in (0,1) for b in (0,1) for c in (0,1)), key=lambda x: ab('hmdb',*x)))} ({f1(max(ab('hmdb',a,b,c) for a in (0,1) for b in (0,1) for c in (0,1)))}% on HMDB51), and on HMDB51 the four configurations with the velocity sampler average {f1(np.mean([ab('hmdb',1,b,c) for b in (0,1) for c in (0,1)]))}% against {f1(np.mean([ab('hmdb',0,b,c) for b in (0,1) for c in (0,1)]))}% for those with uniform sampling. The mechanisms of KAIROS are therefore specific to inputs in which a short interval carries the label; they neither help nor clearly hurt when the evidence is spread over the whole clip."),
          ("tab_comp",),
          ("p", f"Table {{tab:comp}} removes or replaces one element at a time (two seeds, one on SSv2). Without camera compensation TRACE-10 accuracy changes to {f1(P('trace','Ours-noComp'))}%, and without the coverage floor to {f1(P('trace','Ours-rho0'))}%. Context tokens and feature velocity change it to {f1(P('trace','Ours-noCtx'))}% and {f1(P('trace','Ours-noVel'))}% when removed. Replacing the proposed sampler with uniform, MGSampler-style or top-$K$ selection under the same head yields {f1(P('trace','Abl-A0B1C1'))}%, {f1(P('trace','Samp-MG'))}% and {f1(P('trace','Samp-TopK'))}%. The ungated geometry encoding gives {f1(P('trace','Ours-ungated'))}% on TRACE-10 and {f1(P('hmdb','Ours-ungated'))}% on HMDB51, against {f1(P('trace','Ours'))}% and {f1(P('hmdb','Ours'))}% for the gated version, so the validation advantage of the gate on HMDB51 did not carry over to the test split. Three results deserve comment. Camera compensation and feature velocity are the two elements whose removal costs most on TRACE-10 ({f1(P('trace','Ours')-P('trace','Ours-noComp'))} and {f1(P('trace','Ours')-P('trace','Ours-noVel'))} points). Removing the coverage floor raises TRACE-10 accuracy by {f1(P('trace','Ours-rho0')-P('trace','Ours'))} points, which is the price of the floor when all evidence sits in one burst. Context tokens did not help on any dataset; removing them gains {f1(P('trace','Ours-noCtx')-P('trace','Ours'))} points on TRACE-10 and {f1(P('hmdb','Ours-noCtx')-P('hmdb','Ours'))} points on HMDB51. A plausible cause is that the log-mass bias of {{eq:pattn}} lets a context token standing for most of a frame attract a large share of attention, which dilutes the retained patches; this was not tested further."),
          ]
    return B


def sens(ds, name):
    m = {"rho0.2": "Ours", "gamma2.0": "Ours", "beta0.25": "Ours", "K8": "Ours", "rho0.0": "Ours-rho0"}
    n = m.get(name, "Sens-" + name)
    return 100 * R[ds][n]["test"][0] if n == "Ours" else 100 * R[ds][n]["test"][0]


def RESULTS3():
    B = []
    rhos = [0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
    rt = {r: sens("trace", f"rho{r}") for r in rhos}; rh = {r: sens("hmdb", f"rho{r}") for r in rhos}
    br_t = max(rt, key=rt.get); br_h = max(rh, key=rh.get)
    B += [("h2", "D. Sensitivity"),
          ("fig_sens",),
          ("p", f"Fig. {{fig:sensitivity}} varies one setting at a time (seed 0). On TRACE-10, accuracy is {f1(rt[0.0])}% at $\\rho=0$, {f1(rt[0.2])}% at the default $\\rho=0.2$ and {f1(rt[1.0])}% at $\\rho=1$, where the sampler becomes uniform; the best value in the sweep is $\\rho={br_t:g}$. On HMDB51 the curve is flat ({f1(min(rh.values()))}–{f1(max(rh.values()))}%), with its maximum at $\\rho={br_h:g}$, which again indicates that frame choice matters little when evidence is spread over the clip. On TRACE-10 the default $\\gamma=2$ gives the highest accuracy of the three exponents ({f1(sens('trace','gamma1.0'))}%, {f1(sens('trace','gamma2.0'))}% and {f1(sens('trace','gamma3.0'))}% for $\\gamma=1,2,3$). Increasing the retained fraction $\\beta$ from 1/8 to 1 changes TRACE-10 accuracy from {f1(sens('trace','beta0.125'))}% to {f1(sens('trace','beta1.0'))}%, while the STTS-style selector stays between {f1(min(100*R['trace'][n]['test'][0] for n in ['Sens-STTS-beta0.125','Uniform-STTS','Sens-STTS-beta0.5','Sens-STTS-beta1.0']))}% and {f1(max(100*R['trace'][n]['test'][0] for n in ['Sens-STTS-beta0.125','Uniform-STTS','Sens-STTS-beta0.5','Sens-STTS-beta1.0']))}% at every budget, because its frames are sampled uniformly. With $K=4$, 8 and 16 frames KAIROS obtains {f1(sens('trace','K4'))}%, {f1(sens('trace','K8'))}% and {f1(sens('trace','K16'))}% on TRACE-10, against {f1(100*R['trace']['Sens-U-Full-K4']['test'][0])}%, {f1(P('trace','Uniform-Full'))}% and {f1(100*R['trace']['Sens-U-Full-K16']['test'][0])}% for Uniform-Full, so KAIROS with four frames is more accurate than full attention over sixteen uniformly sampled frames. On HMDB51 the same sweep gives {f1(sens('hmdb','K4'))}%, {f1(sens('hmdb','K8'))}% and {f1(sens('hmdb','K16'))}% for KAIROS and {f1(100*R['hmdb']['Sens-U-Full-K4']['test'][0])}%, {f1(P('hmdb','Uniform-Full'))}% and {f1(100*R['hmdb']['Sens-U-Full-K16']['test'][0])}% for Uniform-Full. The values for $\\rho=0$, the defaults and the other baselines at $K=8$ are three- or two-seed means, the rest single runs."),
          ]
    return B


def dil(ds, n, p):
    return 100 * (R[ds][n]["test"][0] if p == 1 else R[ds][n][f"dil{p}"][0])


def RESULTS4():
    B = []
    RB = ["Uniform-CLS", "Uniform-Full", "MG-Full", "Uniform-STTS", "MG-STTS", "Ours"]
    drop = {ds: {n: dil(ds, n, 1) - dil(ds, n, 0.125) for n in RB} for ds in ("hmdb", "ssv2")}
    least = {ds: min(drop[ds], key=drop[ds].get) for ds in drop}
    best8 = {ds: max(RB, key=lambda n: dil(ds, n, 0.125)) for ds in drop}
    B += [("h2", "E. Robustness and Generalization"),
          ("tab_rob",), ("fig_rob",),
          ("p", f"*Idle-stream dilution.* Table {{tab:rob}} and Fig. {{fig:robust}}(a)–(b) compress every test clip into a fraction $\\varphi$ of the lattice and fill the remainder with repeated first or last frames, which imitates an action inside a mostly idle surveillance stream. No model is retrained. At $\\varphi=1/8$ the HMDB51 accuracy of KAIROS falls from {f1(dil('hmdb','Ours',1))}% to {f1(dil('hmdb','Ours',0.125))}%, a drop of {f1(drop['hmdb']['Ours'])} points, while Uniform-Full falls by {f1(drop['hmdb']['Uniform-Full'])} and MG-Full by {f1(drop['hmdb']['MG-Full'])} points; among these six models the smallest drop belongs to {NICE[least['hmdb']]} and the highest accuracy at $\\varphi=1/8$ to {NICE[best8['hmdb']]} ({f1(dil('hmdb',best8['hmdb'],0.125))}%). The TSN-style mean, which ignores frame order, loses {f1(dil('hmdb','TSN-mean',1)-dil('hmdb','TSN-mean',0.125))} points and keeps the highest accuracy overall at $\\varphi=1/8$ ({f1(dil('hmdb','TSN-mean',0.125))}%). On SSv2 the corresponding drops are {f1(drop['ssv2']['Ours'])} points for KAIROS, {f1(drop['ssv2']['Uniform-Full'])} for Uniform-Full and {f1(drop['ssv2']['MG-Full'])} for MG-Full. Motion-driven samplers keep their frames on the action, so they lose less than uniform sampling, but none of them is immune: the action itself is subsampled when it is compressed, and models trained on undiluted clips see a different distribution of frame spacings at test time."),
          ("p", f"*Camera bumps and speed.* On the TRACE-10 test videos that contain a camera bump, KAIROS reaches {f1(100*R['trace']['Ours']['acc_shake'])}% against {f1(100*R['trace']['Ours']['acc_noshake'])}% on videos without one, so the median compensation removes only part of the effect of a bump; for MG + STTS the two figures are {f1(100*R['trace']['MG-STTS']['acc_shake'])}% and {f1(100*R['trace']['MG-STTS']['acc_noshake'])}%. On the two slow classes, which differ from the fast jabs only in duration, KAIROS reaches {f1(100*R['trace']['Ours']['acc_slow'])}% and the strongest baseline on these classes, {NICE[max(MAINB, key=lambda n: R['trace'][n]['acc_slow'])]}, {f1(100*max(R['trace'][n]['acc_slow'] for n in MAINB))}%."),
          ("p", f"*Probe noise.* Adding Gaussian noise with a standard deviation of 0.5 and 1.0 times the mean probe energy changes HMDB51 accuracy of KAIROS to {f1(100*R['hmdb']['Ours']['noise0.5'][0])}% and {f1(100*R['hmdb']['Ours']['noise1.0'][0])}% (TRACE-10: {f1(100*R['trace']['Ours']['noise0.5'][0])}% and {f1(100*R['trace']['Ours']['noise1.0'][0])}%), so the sampler degrades gracefully when the probe signal is corrupted; uniform methods are unaffected by construction."),
          ("p", f"*Frame-count transfer.* Fig. {{fig:robust}}(c) evaluates models trained with $K=8$ at other frame counts without retraining, keeping the token budget fixed. On TRACE-10 KAIROS obtains {f1(100*R['trace']['Ours']['K4'][0])}%, {f1(P('trace','Ours'))}%, {f1(100*R['trace']['Ours']['K12'][0])}% and {f1(100*R['trace']['Ours']['K16'][0])}% at $K=4$, 8, 12 and 16. Accuracy falls when more frames are supplied than during training, because the fixed budget is spread over twice as many frames and the spacing statistics seen by the encoding change. Transfer across frame counts is therefore limited, and a model intended for several budgets should be trained with a randomized $K$."),
          ("h2", "F. Computational Cost"),
          ("tab_eff",),
          ("p", f"Table {{tab:eff}} lists measured costs. One encoder pass of DINOv2 ViT-S/14 at $112\\times112$ costs {FL['enc_gflops_per_frame']:.2f} GFLOPs and {FL['enc_ms_per_frame']:.1f} ms per frame on one CPU thread under bfloat16. The probe costs {FL['probe_mflops_per_frame']:.2f} MFLOPs per raw frame, about {FL['enc_gflops_per_frame']*1e3/FL['probe_mflops_per_frame']:.0f} times less than an encoder pass, but its measured time, {FL['probe_ms_per_frame']:.2f} ms per raw frame, is dominated by resizing the decoded frame and is only about {FL['enc_ms_per_frame']/FL['probe_ms_per_frame']:.0f} times below the encoder time per frame. With roughly {S['data']['hmdb']['nraw_mean']:.0f} raw frames per HMDB51 clip, the probe therefore takes about {S['data']['hmdb']['nraw_mean']*FL['probe_ms_per_frame']:.0f} ms against {8*FL['enc_ms_per_frame']:.0f} ms for eight encoder passes on this CPU. The temporal model of KAIROS costs {FL['heads']['Ours']['gflops']:.3f} GFLOPs and {FL['heads']['Ours']['ms']:.2f} ms, comparable with the other budgeted reducers and about {FL['heads']['Uniform-Full']['gflops']/FL['heads']['Ours']['gflops']:.1f} times cheaper in FLOPs than full attention."),
          ("h2", "G. Calibration"),
          ("tab_cal",),
          ("p", f"Table {{tab:cal}} reports expected calibration error and negative log-likelihood. On TRACE-10 KAIROS has an ECE of {f1(100*R['trace']['Ours']['ece'][0])}% and an NLL of {R['trace']['Ours']['nll'][0]:.2f}; the uniform-frame models are less accurate but also close to chance in confidence, so their ECE ({f1(100*R['trace']['Uniform-Full']['ece'][0])}% for Uniform-Full) is not directly comparable. On HMDB51 the ECE of KAIROS is {f1(100*R['hmdb']['Ours']['ece'][0])}% against {f1(100*R['hmdb']['Uniform-Full']['ece'][0])}% for Uniform-Full, and on SSv2 {f1(100*R['ssv2']['Ours']['ece'][0])}% against {f1(100*R['ssv2']['Uniform-Full']['ece'][0])}%. On the trimmed datasets KAIROS is about one point of ECE less well calibrated than full attention; label smoothing keeps every model below 11% ECE."),
          ]
    return B


def RESULTS5():
    F_ = S["friedman"]
    sig = {ds: [n for n in MAINB if st(ds, n)["p_holm"] < 0.05 and st(ds, n)["delta"] > 0] for ds in DS}
    neg = {ds: [n for n in MAINB if st(ds, n)["p_holm"] < 0.05 and st(ds, n)["delta"] < 0] for ds in DS}
    B = [("h2", "H. Statistical Significance"),
         ("tab_stats",), ("fig_cd",),
         ("p", f"Table {{tab:stats}} reports, for each baseline, the difference to KAIROS with a paired bootstrap confidence interval and a Holm-adjusted $p$-value. On TRACE-10, KAIROS is significantly better than {'all 11' if len(sig['trace']) == 11 else len(sig['trace'])} baselines. On HMDB51 it is significantly better than {'none' if not sig['hmdb'] else len(sig['hmdb'])} and significantly worse than {W.get(len(neg['hmdb']), len(neg['hmdb']))} baselines" + (f" ({', '.join(NICE[n] for n in neg['hmdb'])})" if neg['hmdb'] else "") + ("; on SSv2 no difference is significant after Holm correction" if not (sig['ssv2'] or neg['ssv2']) else f"; on SSv2 it is significantly better than {len(sig['ssv2'])} and worse than {len(neg['ssv2'])}") + f". Across the nine dataset–seed blocks the Friedman test rejects equal ranks ($\\chi^2={F_['chi2']:.1f}$, {pv(F_['p'])}). The critical difference at $\\alpha=0.05$ is {F_['cd']:.2f}. The best mean rank belongs to {NICE[min(F_['mean_rank'], key=F_['mean_rank'].get)]} ({min(F_['mean_rank'].values()):.2f}), and KAIROS has a mean rank of {F_['mean_rank']['Ours']:.2f}; Fig. {{fig:cd}} shows which methods cannot be separated from the best-ranked one. Because the TRACE-10 blocks give KAIROS the first rank by a wide margin while the six trimmed-benchmark blocks give it middle ranks, the rank analysis averages two different regimes and should be read together with the per-dataset intervals."),
         ]
    return B


def DISC():
    pc = json.load(open("results/perclass_hmdb.json"))
    lo = sorted(pc, key=pc.get)[:3]; hi = sorted(pc, key=pc.get)[-3:][::-1]
    nm = lambda s: s.replace("_", " ")
    B = [("h1", "VI. DISCUSSION"),
         ("h2", "A. Why the Method Works Where It Works"),
         ("p", f"The TRACE-10 results follow from a simple accounting argument. With eight uniformly spaced frames and events that cover 5–8% of a video, the expected number of frames inside an event is below one, and a direction of change cannot be read from a single frame. No amount of capacity in the temporal model compensates for evidence that is absent from its input, which is why full attention, every token reducer and even six times more encoder calls on class tokens stay far below KAIROS (at most " + f"{max(P('trace',n) for n in MAINB):.1f}" + "%). The coverage-floored sampler converts the probe signal into frames inside the event in more than nine out of ten videos, the clip-level budget keeps the tokens of those frames, and the gated encoding supplies the elapsed time that distinguishes the slow classes. In the factorial ablation the gain of the full configuration over the base, " + f"{ab('trace',1,1,1)-ab('trace',0,0,0):.1f}" + " points, exceeds the sum of the three single-component gains, " + f"{(ab('trace',1,0,0)+ab('trace',0,1,0)+ab('trace',0,0,1)-3*ab('trace',0,0,0)):.1f}" + " points. The components interact, which is also why the naive combination of a motion-guided sampler with a per-frame token selector (MG + STTS) captures only a fraction of the gain."),
         ("p", "On HMDB51 and on the SSv2 subset the same mechanisms have little to act on. These clips are trimmed around the action, so uniform sampling already covers it, and a frozen image encoder recognizes much of HMDB51 from appearance alone, as the strong TSN-style mean shows. The differences between the methods are then a few points at most. KAIROS is significantly worse than four baselines on HMDB51, by 1.8 to 4.2 points, and indistinguishable from the other seven; on SSv2 none of the differences is significant. This is consistent with the premise of the method, which is temporal sparsity of evidence, and the results give no support to a claim that velocity-proportional allocation improves recognition in general."),
         ("h2", "B. Relation to Prior Findings"),
         ("p", "Motion-guided sampling was reported to help trimmed recognition when the base sampler is fixed [@mgsampler, tmcsampler]; the present MG-Full results agree in direction on TRACE-10 and show no clear benefit on HMDB51. Token reducers such as STTS, ToMe, EViT and run-length tokenization were designed to preserve accuracy while cutting cost on uniformly sampled clips [@stts, tome, rlt], and they do so here as well: at less than a third of the tokens they stay within about three points of full attention on HMDB51. Query-aware samplers for video language models [@aks, bolt] and curvature-based allocation [@vcast, crest] found that moving budget toward transitions matters most for long inputs; the TRACE-10 results reach the same conclusion for a trained recognizer and without query features, and add two elements that these works do not consider, a coverage bound and the encoding of sampling geometry."),
         ("h2", "C. Practical and Methodological Implications"),
         ("p", "For camera networks that must watch long, mostly idle streams, the practical message is that the sampling decision should be made before encoding and from signals that cost almost nothing to compute, and that the downstream model must be told how its frames were spaced. The probe needs no training and runs on decoded thumbnails; in a deployed system it could equally use codec motion vectors [@coviar] or on-camera differencing [@reducto]. The cost analysis also points to a limit: at eight frames and a light temporal model, the encoder dominates, and the saving from token budgeting is a fraction of a GFLOP. The quadratic term in {eq:cost} becomes decisive only for longer horizons or larger temporal models. Fig. {fig:scaling} evaluates {eq:cost} for a 12-layer, 768-wide temporal transformer: at $K=64$ frames, joint attention over all $64\\times17$ tokens would cost about " + f"{EX['scaling']['full'][3]:.0f}" + " GFLOPs, more than the " + f"{EX['scaling']['enc'][3]:.0f}" + " GFLOPs of the frozen encoder, whereas the clip-level budget with $\\beta=1/4$ brings it to " + f"{EX['scaling']['budget'][3]:.0f}" + " GFLOPs. Methodologically, the gating of the geometry terms is a small change with a clear rationale: it lets the same architecture use timing when timing carries the label and ignore it otherwise."),
         ("fig_scaling",),
         ("h2", "D. Failure Cases"),
         ("fig_perclass",),
         ("p", f"Fig. {{fig:perclass}}(a) compares KAIROS with Uniform-Full per class on HMDB51. The largest losses occur for {nm(lo[0])}, {nm(lo[1])} and {nm(lo[2])}, and the largest gains for {nm(hi[0])}, {nm(hi[1])} and {nm(hi[2])}. Classes recognized mainly from a static configuration lose when the sampler concentrates on a short burst of motion and the budget removes background regions that carry context. On TRACE-10 (Fig. {{fig:perclass}}(b)) the remaining errors concentrate on the two tilt classes and on the shrink pulse ({f1(100*R['trace']['Ours']['per_class'][4])}%, {f1(100*R['trace']['Ours']['per_class'][5])}% and {f1(100*R['trace']['Ours']['per_class'][7])}%), where a rotation or a contraction about the object centre changes fewer pixels than a translation and distinguishing clockwise from counter-clockwise needs well-ordered frames inside a short event, and on videos in which the event coincides with a camera bump. Three further failure conditions follow from the design: motion without intensity change (camouflaged or dim objects) is invisible to the probe; global zoom or rotation of the camera is not removed by a median over cells; and actions defined by stillness, such as standing or sitting still, receive no extra frames by construction."),
         ]
    return B


LIMITS = [
 ("h1", "VII. LIMITATIONS"),
 ("p", "*Frozen features and resolution.* All results use a frozen ViT-S/14 at $112\\times112$ and a $4\\times4$ region grid, which keeps the study reproducible on a free Colab instance but places absolute accuracy far below fine-tuned video transformers. Whether the gains on TRACE-10 persist when the encoder is trained end-to-end, or when finer token grids are used, was not tested; this limits external validity rather than internal validity."),
 ("p", "*Benchmarks.* TRACE-10 was built to isolate the failure mode that motivates the method, and the sampler defaults were set on separately generated videos of the same generator. A benchmark designed around a hypothesis can overstate the benefit of a method built for that hypothesis, so the TRACE-10 margin should be read as evidence that the mechanism works when its premise holds, not as an estimate of real-world gain. The SSv2 subset is small, and its absolute numbers are not comparable with full-scale results. A real long-form benchmark with annotated short events, such as untrimmed surveillance footage, is the most important missing evaluation."),
 ("p", "*Candidate lattice and dilution protocol.* Frames are selected from a uniform lattice of $T$ candidates rather than from every raw frame, which caps the temporal resolution inside a burst. The dilution protocol repeats frames exactly, whereas real idle footage contains noise, illumination change and small motions that would raise the probe floor."),
 ("p", "*Cost accounting.* GFLOPs exclude video decoding, which the probe requires for every raw frame. On compressed streams, motion vectors could replace decoding, but that variant was not implemented. Measured CPU latency of the probe is far closer to the encoder cost than its FLOP count suggests."),
 ("p", "*Statistics.* Three seeds are used for the main study and the factorial ablation, two for component removals (one on SSv2) and one for sensitivity sweeps. The single-seed sweeps indicate trends only."),
 ("p", "*Deployment and privacy.* Surveillance analytics raise privacy and proportionality questions that a sampling method does not address. Concentrating computation on moving people may also concentrate errors on them, and any deployment would need an assessment of such effects."),
]


def CONC():
    return [
     ("h1", "VIII. CONCLUSION"),
     ("p", f"Long video streams often carry their label in a short interval, and uniform sampling followed by token reduction cannot recover evidence that never reaches the network. KAIROS addresses this with three coupled mechanisms: a coverage-floored velocity-quantile sampler driven by a camera-compensated pixel probe that runs before any deep encoding, a single token budget shared by frames and regions, and a gated encoding of continuous timestamps, sampling intervals and feature velocity. On TRACE-10, where the label of the fast classes is confined to 5–8% of each video, KAIROS reached {f1(P('trace','Ours'))}% top-1 accuracy against {f1(P('trace',best_base('trace')))}% for the strongest of eleven baselines at the same encoder budget, placed two or more frames inside the event in {100*S['hit']['VQS (KAIROS)']['two']:.0f}% of the videos, and the factorial ablation showed that the three components are complementary. On HMDB51 and on the SSv2 subset, where actions fill the trimmed clips, all methods were within a few points of one another and KAIROS did not improve on the best baselines ({f1(P('hmdb','Ours'))}% against {f1(P('hmdb',best_base('hmdb')))}% and {f1(P('ssv2','Ours'))}% against {f1(P('ssv2',best_base('ssv2')))}%)."),
     ("p", "The scientific contribution is therefore a mechanism whose benefit is confined to a clearly identified regime, together with the bounds and diagnostics needed to recognize that regime. For camera networks that process long, mostly idle streams, the results argue for making the sampling decision before encoding and for telling the recognizer how its frames were spaced. The principal limitation is that the long-form evidence comes from a controlled benchmark and a frozen encoder. The most direct next step is an evaluation on untrimmed surveillance video with annotated short events, using codec motion vectors in place of decoded frames and training with a randomized frame budget."),
    ]


BACK = [
 ("h1", "DATA AVAILABILITY"),
 ("pl", "HMDB51 (split 1) and the Something-Something V2 evaluation subset are publicly available from their original providers and as parquet files on the Hugging Face Hub (mteb/HMDB51 and mteb/SomethingSomethingV2). TRACE-10 is generated deterministically by the released code; no video data were collected for this study."),
 ("h1", "CODE AVAILABILITY"),
 ("pl", "The complete source code, experiment plans, analysis scripts and a Google Colab notebook that reproduces every table and figure are provided with this manuscript and will be released in a public repository with an archived, citable release upon publication."),
 ("h1", "ETHICS STATEMENT"),
 ("pl", "The study uses only publicly released benchmark videos and synthetic videos. No new data involving human participants were collected."),
 ("h1", "CONFLICT OF INTEREST"),
 ("pl", "The authors declare no competing interests."),
]

TITLE = "Velocity-Proportional Frame and Token Allocation with Coverage Guarantees for Recognizing Transient Actions in Long Videos"
KEYWORDS = "Action recognition, adaptive frame sampling, efficient video recognition, long-form video, motion saliency, token pruning, video analytics, vision transformer"


def ABSTRACT():
    bt = best_base("trace")
    return (f"Long surveillance and traffic streams often carry their label in a short burst of motion, yet video recognizers read a fixed number of uniformly spaced frames and then prune tokens only inside those frames, so brief events are frequently never seen. Feature-driven selectors must encode every candidate frame before choosing, motion-driven samplers give no guarantee for the rest of the timeline, and none of them tells the recognizer how its frames were spaced. This paper presents KAIROS, which allocates frames and tokens in proportion to scene velocity. A camera-compensated pixel probe, run before deep encoding, drives a coverage-floored quantile sampler with provable bounds on the largest sampling gap and on the samples that fall inside a burst; the patch tokens of the selected frames compete for one clip-level budget; and a gated encoding supplies continuous timestamps, sampling intervals and feature velocity. With a frozen DINOv2 encoder and eight frames per video, KAIROS was compared with eleven baselines on a controlled long-video benchmark of transient events (TRACE-10), HMDB51 and a Something-Something V2 subset, using three seeds. On TRACE-10 it reached {f1(P('trace','Ours'))}% top-1 accuracy against {f1(P('trace',bt))}% for the best baseline and placed at least two frames inside the event in {100*S['hit']['VQS (KAIROS)']['two']:.0f}% of the test videos, against {100*S['hit']['Uniform']['two']:.0f}% for uniform sampling. On the trimmed benchmarks the methods differed by only a few points and KAIROS did not surpass the strongest baselines ({f1(P('hmdb','Ours'))}% and {f1(P('ssv2','Ours'))}%). Velocity-proportional allocation thus pays off when evidence is temporally sparse, and the bounds and hit-rate diagnostics indicate when that regime applies.")
