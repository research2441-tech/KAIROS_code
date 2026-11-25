from docx_build import build
from refs import R as REFS
from content_front import INTRO, RELATED
from content_method import METHOD, ALG1, ALG2, COMPLEX
from content_protocol import PROTO
from content_tables import NOTATION, RELATED_TAB, impl_table
import re
fill={k:"X" for k in ["hm_trval","hm_te","hm_va","hm_tr","ss_n","ss_tr","ss_va","ss_te","pca_evr"]}
F=lambda bl:[tuple(re.sub("«(\\w+)»",lambda m:fill[m.group(1)],x) if isinstance(x,str) else x for x in b) for b in bl]
blocks=F(INTRO)+F(RELATED[:-1])+[("wide_start",),RELATED_TAB,("wide_end",)]+F(RELATED[-1:])+[("wide_start",),("fig","arch","figs/fig1_architecture.png","Architecture.",7.0),("wide_end",)]
blocks+=F(METHOD[:3])+[NOTATION]+F(METHOD[3:])+[("fig","sampler","figs/fig2_sampler.png","Sampler.",3.45),("alg",ALG1[0],ALG1[1],5040),("alg",ALG2[0],ALG2[1],5040)]+F(COMPLEX)+F(PROTO)+[("tab","data","x",[["a"]],[1000],{}),impl_table("98%")]
r=build(blocks,"/tmp/p1.docx",dict(title="T",authors="A",affil="B",abstract="x",keywords="k"))
blocks+=[("h1","REFERENCES")]+[("ref",f"[{v}] "+REFS[k]) for k,v in sorted(r["cites"].items(),key=lambda x:x[1])]
r=build(blocks,"test_full.docx",dict(title="T",authors="A",affil="B",abstract="x",keywords="k"))
print(r["eq"],r["tab"],len(r["cites"]), set(REFS)-set(r["cites"]))
