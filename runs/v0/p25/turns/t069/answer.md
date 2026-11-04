## reasoning summary

I'm finding that using all the sentences with a weight of 0.3 to 0.5 gives a small robust gain over the baseline, around 0.2 to 0.3 percent, and it should hold up if the labels are mostly right. I'm leaning toward that setting.

I'm weighing whether to scan the 'all' setting over weights of 0.3 and 0.5 combined with lambda values of 0.005 and 0.01. I'm also considering whether writing roughly 400 more sentences in the practice style could add another 0.1 to 0.2 percent, though that would cost a full pass, several code runs, and a lot of output.



Every variant scores about 0.982–0.983 against the 0.980 baseline, so the gain holds. Weighting all my sentences lower matches filtering and stays correct on unseen phrasings; I'm running a small weight × lambda scan next.



## tool_use: run_python

```json
{
 "code": "import json\nfrom scipy.special import erf\nf=np.load('base_feats.npz'); y=f['y']\ndef gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))\ndef LN(x): \n    m=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-m)/np.sqrt(v+1e-12)\nA=json.load(open('my_sents_all.json')); ym=np.array(A['labels'])\nXp=LN(f['cls11']+2*f['mean11'])\nXm=LN(np.load('pass_10/cls_hidden_L11.npy').astype(np.float64)+2*np.load('pass_10/mean_hidden_L11.npy'))\nmu=Xp.mean(0); sd=Xp.std(0)+1e-6\nZp=(Xp-mu)/sd; Zm=(Xm-mu)/sd; d=768\ndef fitw(Ztr,ytr,w,lam=0.01):\n    P=gelu(Ztr@Ztr.T/d); T=np.eye(8)[ytr]; ws=w/w.sum()\n    Pm=ws@P; Tm=ws@T; Pc=P-Pm\n    A_=np.linalg.solve(Pc.T@(Pc*w[:,None])+lam*np.eye(len(Pc)), Pc.T@((T-Tm)*w[:,None]))\n    return lambda Z: (gelu(Z@Ztr.T/d)-Pm)@A_+Tm\ndef cvrun(seed, wm=1.0, lam=0.01):\n    rng=np.random.default_rng(seed); perm=rng.permutation(len(y)); fo=np.zeros(len(y),int); fo[perm]=np.arange(len(y))%5\n    pred=np.zeros(len(y),int); predm=[]\n    for k in range(5):\n        tr=fo!=k; ntr=tr.sum()\n        if wm==0: Ztr=Zp[tr]; ytr=y[tr]; w=np.ones(ntr)\n        else:\n            Ztr=np.vstack([Zp[tr],Zm]); ytr=np.concatenate([y[tr],ym]); w=np.concatenate([np.ones(ntr), wm*np.ones(len(ym))])\n        g=fitw(Ztr,ytr,w,lam); pred[~tr]=np.argmax(g(Zp[~tr]),1)\n    return (pred==y).mean()\nfor wm in [0,0.2,0.3,0.5]:\n    print(wm, [round(np.mean([cvrun(s,wm,lam) for s in range(8)]),4) for lam in [0.003,0.005,0.01]])\n"
}
```

