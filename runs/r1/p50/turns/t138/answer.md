## reasoning summary

The m2048 v2 model comes out at rms 0.106, better than the old 0.1155, with stress values of 16, 22, 30 and 41. The m1536 v1 model is worse, at rms 0.172. I'll write a reusable sim in risk.py to compute P(fail) for this, then check m1792 too.



## message

m2048 v2 gives rms 0.106 and stress flips [16, 22, 30, 41], against [19, 32, 48, 77] for m1536 v1. I'll save the risk simulation as a reusable `risk.py` and run it on both candidates.

## tool_use: run_python

```json
{
 "code": "code = r'''\nimport json\nimport numpy as np\nbl=np.load('base_logits.npy'); ids=json.load(open('all_ids.json'))\ndef sm(z): z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)\ndef f1macro_batch(G,Pd):\n    f=np.zeros(G.shape[0])\n    for c in range(4):\n        tp=((Pd==c)&(G==c)).sum(1); fp=((Pd==c)&(G!=c)).sum(1); fn=((Pd!=c)&(G==c)).sum(1)\n        f+=2*tp/np.maximum(2*tp+fp+fn,1)\n    return f/4\ndef risk(models, ks=(1.0,1.25,1.5), R=6000, N=400, seed=1, allow=4):\n    out={}\n    for k in ks:\n        rng=np.random.default_rng(seed)\n        P=sm(bl/k); cum=P.cumsum(1)\n        idxs=rng.integers(0,1000,(R,N)); u=rng.random((R,N))\n        G=(u[...,None]>cum[idxs]).sum(-1)\n        bp=bl.argmax(1)[idxs]; fb=f1macro_batch(G,bp); ab=(bp==G).sum(1)\n        for name,lg in models.items():\n            ep=(bl/k+lg-bl).argmax(1)[idxs]\n            da=(ep==G).sum(1)-ab; fe=f1macro_batch(G,ep)\n            fa=da<-allow; ff=fe<fb-0.02\n            out[(name,k)]=dict(flips=(ep!=bp).sum(1).mean(), mean_da=da.mean(), p_acc=fa.mean(), p_f1=ff.mean(), p_fail=(fa|ff).mean())\n    return out\ndef show(out):\n    for (n,k),v in out.items(): print('k=%.2f %-12s flips %5.1f  E[da] %5.2f  P(acc) %.3f P(f1) %.3f P(fail) %.3f'%(k,n,v['flips'],v['mean_da'],v['p_acc'],v['p_f1'],v['p_fail']))\n'''\nopen('risk.py','w').write(code)\nexec(code)\nshow(risk({'m1536v1':np.load('lg_pass25.npy'),'m2048v2':np.load('lg_pass30_m2048v2.npy')}))\n"
}
```

