## reasoning summary

I'm checking how much the kernel change actually buys: the steps move from 97.97 down to about 97.8, and run 48 landed at 97.75. Against the linear-exact baseline of 97.64, that's only a gain of around 0.15–0.2%, which is probably within noise. Given moderate implementation risk and the cost of two reads plus a verify pass, I'm weighing whether the expected gain justifies it.

I'm weighing a few bigger options against the modest kernel gain. Stashing M0 looks like about +0.4% over the linear baseline and holds up better under noise, though it means sacrificing some residual dims in layers 0–8 and a few extra passes. I'm also wondering whether pooling logits from several layers into the same stash dims could do even better, so I'd want to check some multi-layer combinations offline first.

I'm considering whether the layer-9 readout could get better pooling by also using layer 10, or by writing a uniform head's mean into stash dims at several layers and summing the results at layer 11. Each stash write would cost one head's residual dims at that layer, and the sacrificed heads would perturb the later layers, so the gain would have to be checked against that.

I'm going to check offline whether pooling means from multiple layers, such as C9 and M9 plus earlier M0 through M8 subsets, adds anything beyond the layer-9 readout.



The ~97.8% readout is only about 0.15–0.2% above the exact linear readout, which is within noise, so I'll check offline whether stashing means from several layers helps more.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nF=np.load('feats.npz'); y=F['y']; FOLDS=np.load('folds_rep.npy')\nG={k:F[k].astype(np.float64) for k in F.files if k!='y'}\nS={k:(v-v.mean(0))/(v.std(0)+1e-9) for k,v in G.items()}\ndef cv_dual(keys,lams,w=None):\n    w=w or [1.0]*len(keys)\n    X=np.concatenate([wi*S[k] for k,wi in zip(keys,w)],1)\n    K=X@X.T\n    res={l:0 for l in lams}\n    for fo in FOLDS:\n        for f in range(5):\n            tr=fo!=f; te=fo==f; Y=np.eye(8)[y[tr]]-1/8; ym=Y.mean(0)\n            Ktr=K[np.ix_(tr,tr)]; n=tr.sum(); H=np.eye(n)-1/n\n            Kc=H@Ktr@H; Kte=(K[np.ix_(te,tr)]-Ktr.mean(0))@H - 0  # approx centering\n            Kte=K[np.ix_(te,tr)]-K[np.ix_(te,tr)].mean(1,keepdims=True)-Ktr.mean(0)[None]+Ktr.mean()\n            for l in lams:\n                a=np.linalg.solve(Kc+l*np.eye(n),Y-ym); res[l]+=((Kte@a+ym).argmax(1)==y[te]).sum()\n    return {l:round(res[l]/len(y)/len(FOLDS),4) for l in lams}\nprint('C9M9', cv_dual(['C9','M9'],[300,1000]))\nprint('+M0', cv_dual(['C9','M9','M0'],[300,1000]))\nfor extra in [['M0','M3'],['M0','M5'],['M0','M7'],['M0','M2','M5'],['M0','M1'],['M0','M3','M6']]:\n    print('+',extra, cv_dual(['C9','M9']+extra,[300,1000,2000]))\nprint('+M0 w.7', cv_dual(['C9','M9','M0'],[300,1000],[1,1,0.7]))\nprint('+M0 w1.4', cv_dual(['C9','M9','M0'],[300,1000],[1,1,1.4]))"
}
```

