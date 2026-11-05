## reasoning summary

With the new data, the reduction is around 12 to 17 percent. At m=1792 the new numbers come out roughly 25 to 30 percent worse than the old m=2048 values, so that setting doesn't match the earlier result.

I'm weighing three options by parameter count against the risk of failing, which would be catastrophic. Under a rough expected-value model, the more aggressive choice comes out ahead, but my failure estimates are very uncertain, so I want better data on the smallest setting before deciding.

I'm planning to run a real pass for the m=1536 setting to measure agreement and logit perturbation, with a matching run for m=1792 as well. That's one pass each, which leaves room within the nine I have. I also want to refine the layer-5 CLS target using edited inputs, since the corrected target gave a meaningful R2 gain at m=1536, though that needs its own probe pass for whichever configuration I end up choosing.

I'm thinking about whether the layer-5 fit can stay robust if the upstream layers shift a little. Refitting on pass 13's activations for the m=1536 candidate should mostly hold, and mixing base and edited inputs in that fit might make the FFN less sensitive to upstream error. I'm also wondering whether other secondary sink tokens, like punctuation, could be a source of the remaining error, since their errors seem to fall within normal token behavior.

I'm considering weighting each token's error by its downstream influence, maybe via how much CLS attends to it in later layers. I'm also thinking the post-LN2 output matters more than the raw FFN output, since LayerNorm amplifies errors on tokens with small variance.

I'm wondering whether a smarter selection could beat the current heuristic. Pruning all at once with one-shot OBS didn't work well, since it ignores that many units are removed together, so I'm considering an iterative version that prunes in chunks and recomputes between steps.

I'm settling on the approach without iterative pruning. The remaining gains look like they come from the data, a layer-5 correction at the final configuration, and picking the right m. I'll keep λ at 100, since it was best for layer 0 and a quick check at 30 to 300 isn't worth it. Next I'm building the m=1536 candidate, using old and new data for layers 0 to 2, old data for layers 3 and 4, and a layer-5 fit on the base CLS inputs.

I'm setting up the layer-5 fit to combine base and edited CLS inputs, with the base target being the corrected residual, and I'll cross-validate to pick λ and the mixing weight.



Held-out errors fell 12–17% at each m, so I'm now refitting layer 5 for the m=1536 candidate on both base and pass-13 edited inputs, and will score it on held-out edited inputs.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nexec(open('lsbuild.py').read())\ndef r2(a,b): return 1-((a-b)**2).sum()/((b-b.mean(0))**2).sum()\ndef LN(x,g,b): mu=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-mu)/np.sqrt(v+1e-5)*g+b\npre='roberta.encoder.layer.5.'\ng2=base(pre+'output.LayerNorm.weight').astype(np.float64); be2=base(pre+'output.LayerNorm.bias').astype(np.float64)\nae=np.load('a5e_1536.npy').astype(np.float64)\nab,W1,b1,U,b2=layer_data(5,base)\nGb=gelu(ab@W1.T+b1); Ge=gelu(ae@W1.T+b1)\nxb=ab+Gb@U.T+b2; hb=LN(xb,g2,be2)\nfolds=np.load('folds.npy'); nf=folds.max()+1\ndef fit_eval(m, lam, wb, we, evalset='e'):\n    he=np.zeros_like(hb); hbb=np.zeros_like(hb)\n    for f in range(nf):\n        tr=folds!=f; te=~tr\n        X=np.vstack([Gb[tr],Ge[tr]]); T=np.vstack([xb[tr]-ab[tr], xb[tr]-ae[tr]])\n        w=np.concatenate([np.full(tr.sum(),wb),np.full(tr.sum(),we)]); keepw=w>0\n        X=X[keepw]; T=T[keepw]; w=w[keepw]; sw=np.sqrt(w)[:,None]\n        mu=(w[:,None]*X).sum(0)/w.sum(); tm=(w[:,None]*T).sum(0)/w.sum()\n        Xc=(X-mu)*sw\n        sc=(U**2).sum(0)*(Xc**2).sum(0)\n        keep=np.sort(np.argsort(-sc)[:m])\n        Xk=Xc[:,keep]; R=(T-tm)*sw-Xk@U[:,keep].T\n        D=np.linalg.solve(Xk.T@Xk+lam*np.eye(m),Xk.T@R).T\n        Un=U[:,keep]+D\n        he[te]=LN(ae[te]+(Ge[te][:,keep]-mu[keep])@Un.T+tm,g2,be2)\n        hbb[te]=LN(ab[te]+(Gb[te][:,keep]-mu[keep])@Un.T+tm,g2,be2)\n    return round(r2(he,hb),4), round(r2(hbb,hb),4)\nfor wb,we in [(1,0),(0,1),(1,1),(0.5,1)]:\n    for lam in (10,30,100):\n        print(wb,we,lam, fit_eval(1536,lam,wb,we))\n"
}
```

