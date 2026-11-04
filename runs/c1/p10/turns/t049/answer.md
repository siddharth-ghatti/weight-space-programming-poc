## tool_use: run_python

```json
{
 "code": "exec(open('krr.py').read()); exec(open('lib.py').read())\nfrom scipy.special import erf\ndef gelu(a): return 0.5*a*(1+erf(a/np.sqrt(2)))\nd=np.load('base_feats.npz'); y=d['y']\nF=lnr(5*d['mean_hidden_L9']+d['cls_hidden_L9'])\ndef cv_gelu(F, y, g, b, lam, seeds=(0,1,2), kind='gelu'):\n    accs=[]\n    for s in seeds:\n        P=np.zeros((len(y),8))\n        for te in folds(len(y),5,s,y):\n            tr=np.setdiff1d(np.arange(len(y)),te)\n            Ztr,Zte=zs(F[tr],F[te]); dd=F.shape[1]\n            T=np.eye(8)[y[tr]]; T=T-T.mean(0)\n            if kind=='gelu':\n                Htr=gelu(g*Ztr@Ztr.T/dd+b); Hte=gelu(g*Zte@Ztr.T/dd+b)\n                mu=Htr.mean(0); Hc=Htr-mu\n                Wt=np.linalg.solve(Hc.T@Hc+lam*np.eye(Hc.shape[1]), Hc.T@T)\n                P[te]=(Hte-mu)@Wt\n            else:\n                Htr=rbf(Ztr,Ztr,g); Hte=rbf(Zte,Ztr,g)\n                al=np.linalg.solve(Htr+lam*np.eye(len(tr)),T); P[te]=Hte@al\n        accs.append((P.argmax(1)==y).mean())\n    return np.mean(accs)\nprint('rbf', cv_gelu(F,y,0.4,None,0.003,kind='rbf'))\nfor g,b in [(2,0),(4,-1),(8,-2),(8,-4),(16,-8)]:\n    for lam in [1e-2,1e-1,1]:\n        print(g,b,lam, round(cv_gelu(F,y,g,b,lam,seeds=(0,)),4))\n"
}
```

