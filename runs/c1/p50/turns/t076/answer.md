## reasoning summary

Layer 2's FFN has a large variance ratio of 0.57, likely driven by a few massive-activation dimensions, yet it still reconstructs easily. Per-layer error relative to the S variance is about 2-3% at m=1024, 1-1.7% at m=1536, and 0.5-0.9% at m=2048. Next I'm working out how those errors add up across the nine layers.

Summing the per-layer errors over nine layers gives roughly 20% at m=1024, but the deviation cost doesn't look like it scales linearly. I'll run m=1024 first to see how much it actually drops, and fall back to m=1536 if it fails. Lambda stays at 100, and the final fit will use all tokens rather than just the training split.



Layer 2's FFN output is large but easy to reconstruct, while layers 6–7 are the hardest, with end-to-end error only measurable after compounding. I'm installing m=1024 pruning on layers 0–8 and reading mean features for 1024 ids next.



## tool_use: run_python

```json
{
 "code": "code=r'''\nimport numpy as np, json\nimport scipy.special as sp\ndef gelu(x): return 0.5*x*(1+sp.erf(x/np.sqrt(2)))\nRES=np.array([83,88,222,254,444,495,506,601])\ndef layer_acts(l, base):\n    emap=json.load(open('expo_map.json')); k=emap[str(l)]\n    Zp=np.load(f'pass_{k}/token_hidden_L{l+1}.npy'); msk=np.load(f'pass_{k}/attention_mask.npy').astype(bool)\n    Z=Zp[msk].astype(np.float64)\n    P=f'bert.encoder.layer.{l}.'\n    X=Z*base(P+'attention.output.LayerNorm.weight')+base(P+'attention.output.LayerNorm.bias')\n    W1=base(P+'intermediate.dense.weight').astype(np.float64); bb1=base(P+'intermediate.dense.bias')\n    W2=base(P+'output.dense.weight').astype(np.float64); bb2=base(P+'output.dense.bias').astype(np.float64)\n    A=gelu(X@W1.T+bb1); F=A@W2.T+bb2\n    return X,A,F,W2,bb2\ndef prune_layer(l, m, base, lam=100.0, sel='energy'):\n    X,A,F,W2,bb2=layer_acts(l, base)\n    en=A.var(0)*(W2**2).sum(0)\n    keep=np.sort(np.argsort(-en)[:m])\n    As=A[:,keep]; mu=As.mean(0); Fm=F.mean(0); Ac=As-mu; Fc=F-Fm; W0=W2[:,keep].T\n    Wn=np.linalg.solve(Ac.T@Ac+lam*np.eye(m), Ac.T@Fc+lam*W0); b=Fm-mu@Wn\n    Wfull=W2.copy(); Wfull[:,keep]=Wn.T\n    if l>=2:\n        Wfull[RES,:]=W2[RES,:]; b[RES]=bb2[RES]\n    return keep, Wfull.astype(np.float32), b.astype(np.float32)\ndef apply_prune(layers, m, base, save_edit, lam=100.0):\n    keepd={}\n    for l in layers:\n        keep,Wf,b=prune_layer(l,m,base,lam)\n        save_edit(f'bert.encoder.layer.{l}.output.dense.weight', Wf)\n        save_edit(f'bert.encoder.layer.{l}.output.dense.bias', b)\n        keepd[l]=keep.tolist()\n    return keepd\n'''\nopen('lib3.py','w').write(code); exec(code)\nkeepd=apply_prune(range(9),1024,base,save_edit)\njson.dump(keepd, open('keep_m1024.json','w'))\nprint(set_structure(layers=list(range(9)), ffn_keep={l:keepd[l] for l in range(9)}))\n"
}
```

