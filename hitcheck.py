import numpy as np
from synth import render, R
from common import probe_energy, lattice, aggregate
import kairos as kz
T,K=48,8; idx=lattice(R,T); rows=[]
for vid in range(30000,30200):
    lab=vid%8
    fr,m=render(vid,lab); e,res=probe_energy(fr)
    Ew,Er=aggregate(e,idx),aggregate(res,idx)
    Et=np.maximum(Er-np.median(Er,0,keepdims=True),0)
    Ewt=np.maximum(Ew-np.median(Ew,0,keepdims=True),0)
    ins=lambda s: ((idx[s]>=m["t0"])&(idx[s]<m["t1"]+4))
    r={}
    r["uni"]=ins(kz.uniform_sample(T,K))
    r["mg"]=ins(kz.quantile_sample(Ew.sum(1),K,0,1))
    for nm,E in [("res",Er),("res+tmed",Et),("raw+tmed",Ewt)]:
        for rho in (0,.2):
            for g in (1,2):
                r[f"{nm} r{rho} g{g}"]=ins(kz.quantile_sample(E.sum(1),K,rho,g))
    rows.append(r)
for k in rows[0]:
    H=np.mean([x[k].mean() for x in rows]); two=np.mean([x[k].sum()>=2 for x in rows])
    print(f"{k:22s} H={H:.3f} >=2: {two:.2f}")
