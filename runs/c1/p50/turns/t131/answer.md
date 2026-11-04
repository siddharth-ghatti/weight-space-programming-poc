## reasoning summary

I need the hidden-state mean over tokens at the embedding layer for the expo ids, so I can compute the expected carry from the RES dims at CLS using the wraw, mu0, and scale values.

I'm checking whether the residual-stream dims at CLS match the expected carry, s·wraw^T(M0−mu0), and I'll test that by comparing per-dimension ratios across samples. Then I'll run fit_layer on L0 with drift 0 vs 1, lambda 30/100/300, and a 48/16 heldout split, comparing err_new. Last, I'll see how far the unpruned FFN on the drifted input lands from base h1, to measure attention drift from the carry-head change.



I'm checking whether the carry term shows up in the RES dims at CLS, then comparing refit settings on a 48/16 sentence split.



## tool_use: run_python

```json
{
 "code": "import json, sys; sys.path.insert(0,'.')\nimport lib5; from lib5 import *\nd=np.load('carry_prov.npz')\nH0=np.load('pass_23/token_hidden_L0.npy').astype(np.float64); msk=np.load('pass_23/attention_mask.npy').astype(bool)\nM0=(H0*msk[...,None]).sum(1)/msk.sum(1)[:,None]\nZp=np.load('pass_25/token_hidden_L1.npy').astype(np.float64)\nZc=LN(Zp[:,0])  # CLS\nexp_c=d['s']*(M0-d['mu0'])@d['wraw']\nr=Zc[:,RES]\nprint('corr per dim', [round(np.corrcoef(r[:,j],exp_c[:,j])[0,1],3) for j in range(8)])\n# per-sample proportionality\nratio=(r*exp_c).sum(1)/(exp_c**2).sum(1); print('ratio mean/cv', ratio.mean(), ratio.std()/ratio.mean())\nprint('resid rel', np.linalg.norm(r-ratio[:,None]*exp_c)/np.linalg.norm(r))\nprint('RES mag CLS', np.abs(r).mean(), 'KEEP mag', np.abs(Zc[:,KEEP]).mean())\nrng=np.random.RandomState(0); tr=np.ones(64,bool); tr[rng.choice(64,16,replace=False)]=False\nfor drift in [0.0,1.0]:\n    for lam in [30,100,300]:\n        keep,Wf,b,info,X=fit_layer(0,25,1024,W,base,lam=lam,drift=drift,tr=tr)\n        print(drift, lam, info)\n"
}
```

