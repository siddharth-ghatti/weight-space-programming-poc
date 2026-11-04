## reasoning summary

I'm weighing whether a different dimension could still yield bigger savings, since five passes remain and there's room for code runs. For now I'll set that aside and build the head for the safe design first. The next step is computing the pooling layer's FFN input x1 exactly for all 1496 items from the base features, via v = C9 + M9 @ O.

I'm carrying the pooling layer's LayerNorm through with eps 1e-12 and no residual rows. Then I'll z-score the 760 non-residual features and compare KRR and GELU heads using 5-fold CV across seeds, with RBF as the baseline on the same folds. The candidates are a single GELU unit per center and a two-unit variant, with the centering handled as a bias shift on the output.



Structure is fixed at k=10, m=3072, and 95,312,648 parameters. Next I'll compute the pooling-layer FFN inputs for all 1496 sentences and compare the three head options on identical folds.



## tool_use: run_python

```json
{
 "code": "exec(open('krr.py').read()); exec(open('lib.py').read())\nfrom scipy.special import erf\ndef gelu(a): return 0.5*a*(1+erf(a/np.sqrt(2)))\nd=np.load('base_feats.npz'); y=d['y']\nO=W('bert.encoder.layer.11.attention.output.dense.weight')\ng1=W('bert.encoder.layer.11.attention.output.LayerNorm.weight'); b1=W('bert.encoder.layer.11.attention.output.LayerNorm.bias')\ndef ln(X,g,b,eps=1e-12):\n    mu=X.mean(1,keepdims=True); v=((X-mu)**2).mean(1,keepdims=True); return (X-mu)/np.sqrt(v+eps)*g+b\nX1=ln(d['cls_hidden_L9'].astype(np.float64)+d['mean_hidden_L9'].astype(np.float64)@O.T.astype(np.float64), g1, b1)\nnp.save('X1_full.npy', X1)\nF=X1[:,KEEP]\nprint('LOO rbf', [round(krr_loo(F,y,g,l),4) for g,l in [(0.2,0.003),(0.4,0.003),(0.4,0.01)]])\nseeds=range(4)\npre=[]\nfor s in seeds:\n    for te in folds(len(y),5,s,y):\n        tr=np.setdiff1d(np.arange(len(y)),te)\n        Ztr,Zte=zs(F[tr],F[te]); dd=F.shape[1]\n        pre.append((s,tr,te,Ztr,Zte,Ztr@Ztr.T/dd,Zte@Ztr.T/dd))\ndef ev(kind, params, lam):\n    res={}\n    for s,tr,te,Ztr,Zte,Gtr,Gte in pre:\n        T=np.eye(8)[y[tr]]; T=T-T.mean(0)\n        if kind=='rbf':\n            Ktr=rbf(Ztr,Ztr,params); Kte=rbf(Zte,Ztr,params)\n            P=Kte@np.linalg.solve(Ktr+lam*np.eye(len(tr)),T)\n        else:\n            Htr=np.hstack([gelu(g*Gtr+b) for g,b in params]); Hte=np.hstack([gelu(g*Gte+b) for g,b in params])\n            mu=Htr.mean(0); Hc=Htr-mu\n            # dual solve (n<p possible)\n            A=Hc@Hc.T+lam*np.eye(len(tr)); al=np.linalg.solve(A,T); P=(Hte-mu)@(Hc.T@al)\n        res.setdefault(s,np.zeros((len(y),8)))[te]=P\n    return np.mean([(v.argmax(1)==y).mean() for v in res.values()])\nprint('rbf .4/.003', ev('rbf',0.4,0.003), 'rbf .2/.003', ev('rbf',0.2,0.003))\nfor prm in [[(1,0)],[(1.5,0)],[(1,-0.3)],[(0.7,0)],[(1,0),(2,-1)],[(1,0),(3,-1.5)]]:\n    print(prm, [round(ev('gelu',prm,l),4) for l in [0.01,0.03,0.1]])\n"
}
```

