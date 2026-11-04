## reasoning summary

I'm checking the cosine similarity of the z-scored vectors, which should sit near 1 for a sentence against itself and lower for different sentences, maybe around 0.1 to 0.3. The sharp settings look weak, since with b=-1 and g=4 the term only goes positive for values above 0.25. The setting g=2, b=0, lam=0.1 scores 98.4 on seed 0, against 98.2 for RBF over three seeds, so I want to explore nearby values.

I'm now setting up a small grid search over the gain, bias, and regularization values, trying three seeds for each combination.



## message

GELU features with smooth settings (g=2, b=0, λ=0.1) reach 98.4 on seed 0, against 98.2 for RBF KRR. Next I'll scan around that point using 3 seeds.

## tool_use: run_python

```json
{
 "code": "exec(open('krr.py').read()); exec(open('lib.py').read())\nfrom scipy.special import erf\ndef gelu(a): return 0.5*a*(1+erf(a/np.sqrt(2)))\nd=np.load('base_feats.npz'); y=d['y']\nF=lnr(5*d['mean_hidden_L9']+d['cls_hidden_L9'])\n# precompute fold kernels once per seed\ndef prep(F,y,seeds):\n    out=[]\n    for s in seeds:\n        for te in folds(len(y),5,s,y):\n            tr=np.setdiff1d(np.arange(len(y)),te)\n            Ztr,Zte=zs(F[tr],F[te]); dd=F.shape[1]\n            out.append((s,tr,te,Ztr@Ztr.T/dd, Zte@Ztr.T/dd))\n    return out\nP0=prep(F,y,(0,1,2))\ndef ev(P0,y,g,b,lam):\n    res={}\n    for s,tr,te,Gtr,Gte in P0:\n        Htr=gelu(g*Gtr+b); Hte=gelu(g*Gte+b)\n        T=np.eye(8)[y[tr]]; T=T-T.mean(0)\n        mu=Htr.mean(0); Hc=Htr-mu\n        Wt=np.linalg.solve(Hc.T@Hc+lam*np.eye(Hc.shape[1]), Hc.T@T)\n        res.setdefault(s,np.zeros((len(y),8)))[te]=(Hte-mu)@Wt\n    return np.mean([(v.argmax(1)==y).mean() for v in res.values()])\nimport itertools\nfor g,b,lam in itertools.product([1,2,3],[-0.5,0,0.5],[0.03,0.1,0.3]):\n    print(g,b,lam, round(ev(P0,y,g,b,lam),4))\n"
}
```

