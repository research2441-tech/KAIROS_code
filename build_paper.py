"""Assembles the manuscript into the IEEE template."""
import json, re
from docx_build import build
from refs import R as REFS
from content_front import INTRO, RELATED
from content_method import METHOD, ALG1, ALG2, COMPLEX
from content_protocol import PROTO
from content_tables import NOTATION, RELATED_TAB, impl_table
import tables_results as TR
import content_results as CR

S = TR.S
d = S["data"]
fill = {"hm_trval": str(d["hmdb"]["tr"] + d["hmdb"]["va"]), "hm_te": str(d["hmdb"]["te"]), "hm_va": str(d["hmdb"]["va"]),
        "hm_tr": str(d["hmdb"]["tr"]), "ss_n": str(d["ssv2"]["N"]), "ss_tr": str(d["ssv2"]["tr"]), "ss_va": str(d["ssv2"]["va"]),
        "ss_te": str(d["ssv2"]["te"]), "pca_evr": f"{100 * min(d[k]['pca_evr'] for k in d):.1f}–{100 * max(d[k]['pca_evr'] for k in d):.1f}%"}


def F(blocks):
    out = []
    for b in blocks:
        b = tuple(re.sub("«(\\w+)»", lambda m: fill[m.group(1)], x) if isinstance(x, str) else x for x in b)
        out.append(b)
    return out


W2 = 10368   # full text width in twips (two columns + gap)
W1 = 5040    # one column
blocks = []
blocks += F(INTRO)
blocks += F(RELATED[:-1])
blocks += [("wide_start",), RELATED_TAB, ("wide_end",)]
blocks += F(RELATED[-1:])
blocks += [("wide_start",), ("fig", "arch", "figs/fig1_architecture.png", CR.CAP["arch"], 7.0), ("wide_end",)]
m = F(METHOD)
# notation table right after the problem formulation paragraph
i = [k for k, b in enumerate(m) if b[0] == "h2" and b[1].startswith("B.")][0]
blocks += m[:i] + [NOTATION] + m[i:]
blocks += [("fig", "sampler", "figs/fig2_sampler.png", CR.CAP["sampler"], 3.45)]
blocks += [("alg", ALG1[0], ALG1[1], W1), ("alg", ALG2[0], ALG2[1], W1)]
blocks += F(COMPLEX)
blocks += F(PROTO[:2]) + [("tab", "data", "Datasets and partitions", TR.data_table(), [1350, 1050, 1300, 1340], {"upper": True})]
blocks += F(PROTO[2:])
blocks += [impl_table(fill["pca_evr"])]
PH = {
 "tab_main": [("wide_start",), ("tab", "main", "Main comparison at a matched encoder budget (top-1 accuracy and macro-F1 in %, mean ± standard deviation over three seeds; best per column in bold)", TR.main_table(), [1900, 620, 760, 820, 1250, 1250, 1250, 1250, 1268], {"upper": True, "size": 13}), ("wide_end",)],
 "tab_hit": [("tab", "hit", "Burst hit rate of the samplers on the TRACE-10 test videos ($K=8$)", TR.hit_table(), [1500, 700, 950, 950, 940], {"upper": True})],
 "fig_tokens": [("wide_start",), ("fig", "tokens", "figs/fig3_tokens.png", CR.CAP["tokens"], 7.0), ("wide_end",)],
 "tab_abl": [("tab", "abl", "Factorial ablation of the sampler (A), cross-frame budget (B) and geometry encoding (C), top-1 in %", TR.abl_table(), [360, 360, 360, 1320, 1320, 1320], {"upper": True})],
 "fig_ablation": [("fig", "ablation", "figs/fig5_ablation.png", CR.CAP["ablation"], 3.45)],
 "tab_comp": [("tab", "comp", "Component removals and sampler swaps under the full head, top-1 in % (two seeds; one on SSv2)", TR.comp_table(), [1860, 1060, 1060, 1060], {"upper": True})],
 "fig_sens": [("wide_start",), ("fig", "sensitivity", "figs/fig6_sensitivity.png", CR.CAP["sensitivity"], 7.0), ("wide_end",)],
 "tab_rob": [("wide_start",), ("tab", "rob", "Robustness: idle-stream dilution, probe noise and frame-count transfer (top-1 in %, mean over three seeds)", TR.robust_table(), [1700] + [775] * 11, {"upper": True, "head": 2, "size": 13}), ("wide_end",)],
 "fig_rob": [("wide_start",), ("fig", "robust", "figs/fig7_robustness.png", CR.CAP["robust"], 7.0), ("wide_end",)],
 "tab_eff": [("tab", "eff", "Measured cost per video on one CPU thread (HMDB51 clip length)", TR.eff_table(), [1180, 620, 600, 700, 700, 640, 600], {"upper": True, "size": 13})],
 "tab_cal": [("tab", "cal", "Calibration on the test splits (mean over three seeds)", TR.calib_table(), [1440, 600, 600, 600, 600, 600, 600], {"upper": True, "head": 2, "size": 13})],
 "tab_stats": [("wide_start",), ("tab", "stats", "Difference between KAIROS and each baseline: paired bootstrap 95% confidence interval, Holm-adjusted p-value per dataset, and Wilcoxon signed-rank p-value over the nine dataset–seed blocks", TR.stats_table(), [1700, 1450, 780, 1450, 780, 1450, 780, 1000], {"upper": True, "head": 2, "size": 13}), ("wide_end",)],
 "fig_hitdur": [("fig", "hitdur", "figs/fig_hitdur.png", CR.CAP["hitdur"], 3.45)],
 "fig_scaling": [("fig", "scaling", "figs/fig_scaling.png", CR.CAP["scaling"], 3.45)],
 "tab_contrib": [("tab", "contrib", "Contribution-to-evidence map", [["Contribution", "Method", "Evidence"],
     ["1) Coverage-floored velocity-quantile sampler", "Sec. III-B, III-C; {eq:cell}–{eq:bound}", "Tables {tab:hit}, {tab:comp}; Figs. {fig:sampler}, {fig:hitdur}, {fig:sensitivity}"],
     ["2) Cross-frame token budget", "Sec. III-D; {eq:score}–{eq:pattn}", "Tables {tab:abl}, {tab:comp}; Figs. {fig:tokens}, {fig:ablation}"],
     ["3) Gated sampling-geometry encoding", "Sec. III-E; {eq:four}–{eq:emb}", "Tables {tab:abl}, {tab:comp}, {tab:rob}; Fig. {fig:robust}"],
     ["4) Evaluation and regime of benefit", "Sec. IV, V", "Tables {tab:main}, {tab:eff}, {tab:cal}, {tab:stats}; Figs. {fig:pareto}, {fig:cd}, {fig:perclass}"]],
     [1700, 1500, 1840], {"upper": True})],
 "fig_cd": [("fig", "cd", "figs/fig8_cd.png", CR.CAP["cd"], 3.45)],
 "fig_perclass": [("wide_start",), ("fig", "perclass", "figs/fig9_perclass.png", CR.CAP["perclass"], 7.0), ("wide_end",)],
}
def expand(bl):
    out = []
    for b in bl:
        out += PH[b[0]] if b[0] in PH else [b]
    return out
blocks += [("fig", "pareto", "figs/fig4_pareto.png", CR.CAP["pareto"], 3.45)] if False else []
res_blocks = CR.RESULTS(TR) + CR.RESULTS2() + CR.RESULTS3() + CR.RESULTS4() + CR.RESULTS5()
# Pareto figure after the main-comparison paragraphs
k = [i for i, b in enumerate(res_blocks) if b[0] == "h2" and b[1].startswith("B.")][0]
res_blocks = res_blocks[:k] + [("wide_start",), ("fig", "pareto", "figs/fig4_pareto.png", CR.CAP["pareto"], 7.0), ("wide_end",)] + res_blocks[k:]
blocks += expand(res_blocks)
blocks += expand(CR.DISC())
blocks += CR.LIMITS
blocks += CR.CONC()
blocks += CR.BACK
blocks = expand(blocks)
blocks += [("h1", "REFERENCES")]

res = None


def refs_blocks(cites):
    order = sorted(cites, key=lambda k: cites[k])
    return [("ref", f"[{cites[k]}] " + REFS[k]) for k in order]


# two passes: first to obtain citation order, then with the reference list
FRONT = dict(CR.FRONT, title=CR.TITLE, abstract=CR.ABSTRACT(), keywords=CR.KEYWORDS)
res = build(blocks, "/tmp/pass1.docx", FRONT)
missing = set(REFS) - set(res["cites"]); extra = set(res["cites"]) - set(REFS)
assert not extra, extra
print("uncited:", missing)
res = build(blocks + refs_blocks(res["cites"]), "KAIROS_manuscript.docx", FRONT)
print(res["eq"], "equations", res["fig"], "figures", res["tab"], "tables", len(res["cites"]), "references")
