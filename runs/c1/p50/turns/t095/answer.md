## reasoning summary

I'm checking whether the current model really matches the sequential m=1024 fit, since the layer-l exposures would have been taken with layers 0 through l-1 already pruned. I think they were, because the earlier passes only applied the final edits from the lower layers.

I'm now measuring the current drift in pre-LN space by running the layer's FFN on the exposures with the saved edits, and comparing the output to the base hidden state. Then I'll compare held-out error for the current fit against a drift-corrected fit to see the per-layer gain. I also need to confirm the exposures share the same sequence length, T=63, so the masks line up.



I can now measure, layer by layer, how far the pruned layer's output drifts from the base hidden state, and whether a drift-corrected refit tracks base better on held-out sentences.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0, str(WORK))\nfrom lib3 import *\nfrom lib import KEEP\nemap=json.load(open('expo_map.json'))\nmsk20=np.load('pass_20/attention_mask.npy').astype(bool)\nkeepd=structure()['ffn_keep']\ndef LN(u,g,b):\n    mu=u.mean(1,keepdims=True); sd=np.sqrt(u.var(1,keepdims=True)+1e-12); return (u-mu)/sd*g+b\ndef rel(a,b): return np.linalg.norm(a-b)/np.linalg.norm(b-b.mean(0))\nres={}\nfor l in range(9):\n    k=emap[str(l)]; msk=np.load(f'pass_{k}/attention_mask.npy').astype(bool); assert (msk==msk20).all()\n    X,A,F,W2,bb2=layer_acts(l, base)\n    P=f'bert.encoder.layer.{l}.'\n    g=base(P+'output.LayerNorm.weight').astype(np.float64); bt=base(P+'output.LayerNorm.bias').astype(np.float64)\n    Hb=np.load(f'pass_20/token_hidden_L{l+1}.npy')[msk20].astype(np.float64)\n    kept=np.array(keepd[str(l)] if isinstance(keepd,dict) and str(l) in keepd else keepd[l])\n    Wc=W(P+'output.dense.weight').astype(np.float64); bc=W(P+'output.dense.bias').astype(np.float64)\n    Fc=A[:,kept]@Wc[:,kept].T+bc\n    sid=np.repeat(np.arange(64)[:,None],63,1)[msk20]; te=sid>=32; tr=~te\n    full=LN(X+F,g,bt); cur=LN(X+Fc,g,bt)\n    # drift-corrected target\n    U=X+F; s=U.std(1,keepdims=True); m_=U.mean(1,keepdims=True)\n    gs=np.where(np.abs(g)<1e-3,1e-3,g)\n    T=(Hb-bt)/gs*s+m_-X\n    lam=100.0; As=A[:,kept]; mu=As[tr].mean(0); Ac=As[tr]-mu; W0=W2[:,kept].T\n    G=Ac.T@Ac+lam*np.eye(len(kept))\n    out=[]\n    for alpha in [0.0,0.5,1.0]:\n        Tg=alpha*T+(1-alpha)*F; Tm=Tg[tr].mean(0)\n        Wn=np.linalg.solve(G, Ac.T@(Tg[tr]-Tm)+lam*W0); b=Tm-mu@Wn\n        Fn=As@Wn+b\n        if l>=2: Fn[:,RES]=F[:,RES]\n        o=LN(X+Fn,g,bt); out.append((alpha, rel(o[te][:,KEEP],Hb[te][:,KEEP])))\n    print(l,'fullFFN-vs-base %.3f  cur-vs-base %.3f'%(rel(full[te][:,KEEP],Hb[te][:,KEEP]), rel(cur[te][:,KEEP],Hb[te][:,KEEP])), ' new:',' '.join('a%.1f:%.3f'%x for x in out))"
}
```

