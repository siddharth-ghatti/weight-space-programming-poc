## reasoning summary

I'm settling the pool layout: the first 1008 units are anchors with row norm 1/√768, and the last 16 are linear pairs that read one residual dimension and write back through an 8×8 map B. So the deployed pool gives F[RES] as the anchor term plus B applied to the residual, plus a bias, and the final hidden state is LN(z + F).

I'm deciding how to refit the linear block: the 8 carried scores are z[RES] read from the edited model, and I'd jointly refit them with the anchors, possibly with a penalty, then install the resulting coefficients and check the final LN(z+F) decode.

I'm checking whether the decode's per-sample scale and mean shift can change the classifier's argmax; since the pooler centers and the scale is positive, it looks like they can't, so the residual term's effect is what remains to be confirmed.

I'm checking whether the residual pass-through changes the fit: the scores come out as z[RES] + F[RES], so the effective linear block is I + B, and installing it would mean subtracting I from the fitted coefficients. I could also leave the identity term in and fit against the full sum instead. Next I want to rebuild the earlier fit from the saved prediction file to confirm which convention it used.

I'm checking the pool-layer forward pass against the saved prediction file, computing LN, pooler, and tanh from the features to confirm the fit's convention. I'll use this only for fitting and verifying the closed-form pool, as before.

I'm checking whether the bias recovers the same mean shift as the mean of the 1496 depth-M9 features, which is what the fit used.



I'm rebuilding the installed pool's numpy prediction from the old 10-layer features to confirm how c, mu, the residual pass-through and the decode were handled before refitting.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nimport numpy as np\nfrom poolfit import *\nP='bert.encoder.layer.11.'\nWo=W(P+'attention.output.dense.weight'); bo=W(P+'attention.output.dense.bias')\ndg=np.diag(Wo).astype(np.float64)\nprint('diag KEEP unique', np.unique(dg[KEEP].round(4))[:5], 'RES', dg[RES])\nmu_inst=-bo/dg\nf=np.load('feats_seq1024_depth.npz'); M9=f['M9'].astype(np.float64); C9=f['C9'].astype(np.float64)\nprint('mu_inst vs M9 mean', np.abs(mu_inst-M9.mean(0)).max(), np.abs(mu_inst).max())\nW1=W(P+'intermediate.dense.weight').astype(np.float64)[:1024]; b1=W(P+'intermediate.dense.bias').astype(np.float64)[:1024]\nW2=W(P+'output.dense.weight').astype(np.float64)[:, :1024]; b2=W(P+'output.dense.bias').astype(np.float64)\npw=W('bert.pooler.dense.weight').astype(np.float64)[:8][:,RES]; pb=W('bert.pooler.dense.bias').astype(np.float64)[:8]\nprint('pooler 8x8', np.round(pw,3)); \nx=dg*M9+bo+C9; z=LN(x)\nF=gelu(z@W1.T+b1)@W2.T+b2\nh=LN(z+F)\nlogit=np.tanh(h[:,RES]@pw.T+pb)\nref=np.load('pred_logits_pool1.npy'); print('max diff vs pred_logits_pool1', np.abs(logit-ref).max())\nlg36=np.load('pass_36/logits.npy'); print('vs pass36', np.abs(logit[:1024]-lg36).max())\nprint('lin pair W1 rows', [(int(np.nonzero(W1[u])[0][0]), W1[u][np.nonzero(W1[u])[0][0]]) for u in range(1008,1024)])"
}
```

