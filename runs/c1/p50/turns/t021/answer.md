## reasoning summary

I'm comparing the first 1024 results against the base, using per-layer linear CV on mean and cls tokens plus RBF combinations. I'll also write a compare function in lib that takes a pass directory and a reference.



I'm comparing experiment A against the unpruned base on the same 1,024 sentences, using per-depth linear CV and a combined RBF score, then checking representation drift against base.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nadd = r'''\ndef krr_cv(K, y, lams, k=5, seeds=(0,1)):\n    out={}; Y=np.eye(8)[y]\n    for lam in lams:\n        accs=[]\n        for seed in seeds:\n            f=folds(y,k,seed); S=np.zeros((len(y),8))\n            for i in range(k):\n                tr=f!=i; te=f==i\n                A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0)); S[te]=K[np.ix_(te,tr)]@A\n            accs.append((S.argmax(1)==y).mean())\n        out[lam]=round(float(np.mean(accs)),4)\n    return out\ndef rbf(X,gf=1):\n    Xs=X-X.mean(0); sq=(Xs**2).sum(1); D2=sq[:,None]+sq[None,:]-2*Xs@Xs.T\n    return np.exp(-D2/(gf*np.median(D2)))\ndef blockn(Xs): return np.concatenate([(x-x.mean(0))/np.sqrt(((x-x.mean(0))**2).sum(1).mean()) for x in Xs],1)\ndef loadpass(k, n=None):\n    import glob,os\n    out={}\n    for f in glob.glob(f'pass_{k}/*.npy'):\n        out[os.path.basename(f)[:-4]]=np.load(f)\n    return out\n'''\nopen('lib.py','a').write(add)\nfrom lib import *\ny=labels()[:1024]; d=load_base(); e=loadpass(5)\nMb=lambda l: d[f'mean_hidden_L{l}'][:1024][:,KEEP]; Cb=lambda l: d[f'cls_hidden_L{l}'][:1024][:,KEEP]\nMe=lambda l: e[f'mean_hidden_L{l}'][:,KEEP]; Ce=lambda l: e[f'cls_hidden_L{l}'][:,KEEP]\nfor l in [1,2,4,6,8,10,11]:\n    rb=cv_acc(Mb(l),y,lams=(3,10,30)); re_=cv_acc(Me(l),y,lams=(3,10,30))\n    cos=np.mean(np.sum(Mb(l)*Me(l),1)/np.linalg.norm(Mb(l),axis=1)/np.linalg.norm(Me(l),axis=1))\n    print(l,'base',max(rb.values()),'pruned',max(re_.values()),'cos',round(cos,3))\nfor j in [6,8,11]:\n    sel=[0,j//2,j]\n    print(j,'rbf base',krr_cv(rbf(blockn([Mb(l) for l in sel]+[Cb(j)])),y,(0.003,0.01)),'pruned',krr_cv(rbf(blockn([Me(l) for l in sel]+[Ce(j)])),y,(0.003,0.01)))"
}
```

