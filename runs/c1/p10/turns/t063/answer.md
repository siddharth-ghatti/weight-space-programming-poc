## reasoning summary

I'm comparing the transductive LOO numbers, which look artificially low at small lambda because the held-out point's own feature column leaks into the centers. The non-transductive 5-fold results at lambda 0.01 to 0.03 sit around 98.2 to 98.3, and the RBF kernel ridge baseline is 98.66 under LOO and 98.23 under 5-fold. I'm now settling on GELU with g=1 and b=0 as the choice.

I'm weighing the two lam=0.03 and lam=0.01 options, since their 5-fold and LOO numbers point in different directions. The transductive LOO isn't a reliable guide for a deployed model where the test point isn't a center, so I'm switching to 20-fold non-transductive CV over a grid of lam and g values, with RBF as a reference.



Transductive LOO is unreliable at small λ and test points are never centres at deployment, so I'm switching to non-transductive 20-fold CV (2 seeds) to pick g and λ, with RBF as the reference.



## tool_use: run_python

```json
{
 "code": "exec(open('krr.py').read()); exec(open('lib.py').read())\nfrom scipy.special import erf\ndef gelu(a): return 0.5*a*(1+erf(a/np.sqrt(2)))\ny=np.load('base_feats.npz')['y']; X1=np.load('X1_full.npy'); F=X1[:,KEEP]\npre=[]\nfor s in (0,1):\n    for te in folds(len(y),20,s,y):\n        tr=np.setdiff1d(np.arange(len(y)),te)\n        Ztr,Zte=zs(F[tr],F[te]); dd=F.shape[1]\n        pre.append((s,tr,te,Ztr,Zte,Ztr@Ztr.T/dd,Zte@Ztr.T/dd))\ndef ev(kind, g, b, lams):\n    out={l:{} for l in lams}\n    for s,tr,te,Ztr,Zte,Gtr,Gte in pre:\n        T=np.eye(8)[y[tr]]; T=T-T.mean(0)\n        if kind=='rbf':\n            Ktr=rbf(Ztr,Ztr,g); Kte=rbf(Zte,Ztr,g)\n            w,V=np.linalg.eigh(Ktr); VT=V.T@T\n            for l in lams: out[l].setdefault(s,np.zeros((len(y),8)))[te]=Kte@(V@(VT/(w+l)[:,None]))\n        else:\n            Htr=gelu(g*Gtr+b); Hte=gelu(g*Gte+b); mu=Htr.mean(0); Hc=Htr-mu\n            w,V=np.linalg.eigh(Hc@Hc.T); VT=V.T@T; Hx=(Hte-mu)@Hc.T\n            for l in lams: out[l].setdefault(s,np.zeros((len(y),8)))[te]=Hx@(V@(VT/(w+l)[:,None]))\n    return {l:round(np.mean([(v.argmax(1)==y).mean() for v in out[l].values()]),4) for l in lams}\nprint('rbf .4', ev('rbf',0.4,0,[0.001,0.003,0.01,0.03]))\nfor g,b in [(0.7,0),(1,0),(1.3,0),(1,0.3),(1,-0.3)]:\n    print('gelu',g,b, ev('gelu',g,b,[0.003,0.01,0.02,0.03,0.05,0.1]))\n"
}
```

