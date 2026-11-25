import torch, timm, time
torch.set_num_threads(2)
for name,sz in [("vit_small_patch14_dinov2.lvd142m",112),("vit_small_patch14_dinov2.lvd142m",126)]:
    m=timm.create_model(name,pretrained=True,img_size=sz,num_classes=0).eval()
    x=torch.randn(32,3,sz,sz)
    with torch.inference_mode():
        m.forward_features(x)
        t=time.time(); 
        for _ in range(3): f=m.forward_features(x)
        dt=(time.time()-t)/96
    print(name,sz,f.shape, f"{1/dt:.1f} img/s")
