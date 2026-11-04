## reasoning summary

I'm going to make each run a fresh process, so I'll put the shared kernel helpers in a separate file called kh.py.



## message

Each run starts a fresh process, so I'll put the kernel helpers in a file and re-run the CV.

## tool_use: run_python

```json
{
 "code": "open('kh.py','w').write('''\nimport numpy as np\nfrom scipy.special import erf\nfrom helpers import macro_f1\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\ndef fit(Phi,y,w,lam):\n    T=np.eye(8)[y]-1/8\n    sw=w/w.sum(); m=sw@Phi; tm=sw@T\n    Pc=Phi-m; Tc=T-tm\n    G=(Pc*w[:,None]).T@Pc+lam*np.eye(Phi.shape[1])\n    a=np.linalg.solve(G,(Pc*w[:,None]).T@Tc)\n    return a, tm-m@a+1/8\ndef fitk(Ztr,ytr,wtr,lam,beta=1.0):\n    P=gelu(beta*Ztr@Ztr.T/768); return fit(P,ytr,wtr,lam)\ndef load():\n    k=np.load('kernel_v3.npz'); C,yc=k['C'],k['yc']\n    d3=np.load('my3_feats.npz'); d4=np.load('my4_feats.npz')\n    return dict(Zp=C[:1496],yp=yc[:1496],Zm=C[1496:],ym=yc[1496:],Z3=d3['Z'],y3=d3['y'],Z4=d4['Z'],y4=d4['y'],mu=k['mu'],sd=k['sd'])\ndef folds(yp,s,nf=5):\n    rng=np.random.default_rng(s); fo=np.zeros(len(yp),int)\n    for c in range(8):\n        idx=np.where(yp==c)[0]; rng.shuffle(idx); fo[idx]=np.arange(len(idx))%nf\n    return fo\ndef run_cv(D, extra, wx, lam, seeds=(0,1,2,3), nf=5, beta=1.0):\n    Zp,yp=D['Zp'],D['yp']; accs=[];f1s=[]\n    for s in seeds:\n        fo=folds(yp,s,nf); pred=np.zeros(len(yp),int)\n        for f in range(nf):\n            tr=fo!=f\n            if np.isscalar(wx): wl=[wx]*len(extra)\n            else: wl=wx\n            Zt=np.concatenate([Zp[tr]]+[e[0] for e in extra]); yt=np.concatenate([yp[tr]]+[e[1] for e in extra])\n            wt=np.concatenate([np.ones(tr.sum())]+[np.full(len(e[1]),ww) for e,ww in zip(extra,wl)])\n            a,b=fitk(Zt,yt,wt,lam,beta)\n            pred[~tr]=np.argmax(gelu(beta*Zp[~tr]@Zt.T/768)@a+b,1)\n        accs.append((pred==yp).mean()); f1s.append(macro_f1(yp,pred))\n    return round(np.mean(accs),4), round(np.mean(f1s),4), np.round(accs,4)\n''')\nfrom kh import *\nimport time; t=time.time()\nD=load()\nE={'m':(D['Zm'],D['ym']),'m3':(D['Z3'],D['y3']),'m4':(D['Z4'],D['y4'])}\nfor name,ex,wx,lam in [('prac',[],0.3,0.01),('+m',['m'],0.3,0.005),('+m3',['m3'],0.3,0.03),('+m3m4',['m3','m4'],0.3,0.03),('+all',['m','m3','m4'],0.3,0.03)]:\n    print(name,wx,lam,run_cv(D,[E[e] for e in ex],wx,lam), round(time.time()-t))\n"
}
```

