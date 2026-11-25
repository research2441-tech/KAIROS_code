import torch, time
from kairos import Head
torch.set_num_threads(1)
B,K,D=64,8,192
X=torch.randn(B,K,17,D); M=torch.rand(B,K,16); tau=torch.sort(torch.rand(B,K),1).values
for red,b,enc in [("cls",0,"index"),("full",0,"index"),("stts",32,"index"),("tome",32,"index"),("evit",24,"index"),("rlt",32,"index"),("ours",24,"ct"),("random",32,"index")]:
    m=Head(D,51,dict(reducer=red,B=b,enc=enc)); opt=torch.optim.AdamW(m.parameters())
    t=time.time()
    for _ in range(3):
        loss=m(X,M,tau).logsumexp(-1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    print(red, f"{(time.time()-t)/3:.3f}s/batch64")
