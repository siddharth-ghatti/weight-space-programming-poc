## reasoning summary

The filter only gained 0.36% over all sentences, but it's built on a model that saw the test fold's labels, so that small gain might just come from leakage. My sentences that agree with the full-practice model would tend to reinforce the held-out data, which would inflate the result.

The leak question is still open: the filter uses a model that saw the held-out fold, so the gain might come from that. I'm going to redo the filter per fold, computing agreement only with a model trained on the training-fold practice data. Separately, about 45 of my sentences disagree with the model. I want to check whether those are labeling mistakes on my side or real model errors, since accuracy on my sentences is lower than on the in-distribution data.



Adding all sentences gave no gain, and filtering with a practice model reached 0.985, though that model saw the held-out folds, so it may be optimistic. Next I'll redo the filtering inside each fold and examine the disagreements.



## tool_use: run_python

```json
{
 "code": "import json\nfrom scipy.special import erf\nf=np.load('base_feats.npz'); y=f['y']\ndef gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))\ndef LN(x): \n    m=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-m)/np.sqrt(v+1e-12)\nA=json.load(open('my_sents_all.json')); ym=np.array(A['labels'])\nXp=LN(f['cls11']+2*f['mean11'])\nXm=LN(np.load('pass_10/cls_hidden_L11.npy').astype(np.float64)+2*np.load('pass_10/mean_hidden_L11.npy'))\nmu=Xp.mean(0); sd=Xp.std(0)+1e-6\nZp=(Xp-mu)/sd; Zm=(Xm-mu)/sd; d=768\ndef fit(Ztr,ytr,lam=0.01):\n    P=gelu(Ztr@Ztr.T/d); T=np.eye(8)[ytr]; Pm=P.mean(0); Tm=T.mean(0); Pc=P-Pm\n    A_=np.linalg.solve(Pc.T@Pc+lam*np.eye(len(Pc)), Pc.T@(T-Tm))\n    return lambda Z: (gelu(Z@Ztr.T/d)-Pm)@A_+Tm\ndef cvrun(seed, mode, lam=0.01, thr=None):\n    rng=np.random.default_rng(seed); perm=rng.permutation(len(y)); fo=np.zeros(len(y),int); fo[perm]=np.arange(len(y))%5\n    pred=np.zeros(len(y),int)\n    for k in range(5):\n        tr=fo!=k\n        g0=fit(Zp[tr],y[tr],lam)\n        if mode=='none': g=g0\n        else:\n            S=g0(Zm); pmk=np.argmax(S,1)\n            mm = pmk==ym\n            if mode=='relabel':\n                Ztr=np.vstack([Zp[tr],Zm]); ytr=np.concatenate([y[tr],pmk])\n            else:\n                Ztr=np.vstack([Zp[tr],Zm[mm]]); ytr=np.concatenate([y[tr],ym[mm]])\n            g=fit(Ztr,ytr,lam)\n        pred[~tr]=np.argmax(g(Zp[~tr]),1)\n    return (pred==y).mean()\nfor mode in ['none','agree','relabel']:\n    r=[cvrun(s,mode) for s in range(8)]\n    print(mode, round(np.mean(r),4), np.round(r,4))\n"
}
```

