## reasoning summary

I'm checking whether the small mean shifts (about 0.03) explain the flips, which seem to mostly move items out of the strategy class. The base model underpredicts strategy in-sample, and its errors are mostly strategy items predicted as something else, so the low-margin cases are where noise pushes them away from strategy. I'm now trying to pin down why the flips happen.

I'm now weighing whether the in-sample flips are systematic: the pruned model seems to drift away from the base in a "less fine-tuned" direction, which could hurt on training items but shouldn't carry over to the test set, where flips should be roughly symmetric. Still, I can't verify that, so I'm starting to put together a rough risk estimate for the test accuracy.

I'm estimating the test-set risk now: the base model is at 77.5% on test, so I need at least 306 of 400 correct. Noise std varies by configuration, and I'm guessing the test set has roughly 15% of items under margin 0.5 and another 15% between 0.5 and 1. Working through the flip counts under that assumption, the m1536 configuration shows about 22 flips and m2048 about 14, and I'm now checking what net loss that implies given how many flips break correct predictions versus fix wrong ones.

I'm working out the expected net loss on test: about 5.2 for the m1536 configuration, giving roughly a 40% chance of staying within the 4-item slack. The m2048 configuration does better, with about a 57% chance, though that is still a substantial failure risk given how little margin the base model leaves.

I'm weighing how much the variance alone threatens the test threshold: with symmetric flips, the risk is roughly 14% for m2048, 20% for m1536, and 25% for m1024. The macro-F1 slack of about 0.02 is also tight, since a couple of governance flips could move it that much.

I'm weighing whether to accept some in-sample pessimism, since the flips might be biased toward hurting on training items but roughly symmetric on test. The core tradeoff is compression size against risk, and I'd want the chance of passing to stay at or above 80–85%. I'm also considering whether layer-level compensation or dropping a layer to allow a larger margin could cut the noise further.

I'm thinking about whether a tighter in-sample fidelity target and a margin of about 1536–2048 could work, given the new data cut compensation errors by roughly 10–15%. Before I can judge the tradeoff, I need the agreement numbers at those margins from fresh passes over layers 0–4.

I'm planning to rebuild layers 0 through 4 at the m=1536 margin using the new data, with the layer 5 CLS samples included, then read the logits in one pass and compare against the earlier 0.977 agreement. If it helps, I might run m=2048 as well, and I'm also considering a sequential correction for the token layers that uses the base targets.

I'm considering whether a smarter allocation could help, since the error from layers 1 and 2 is larger than the others but I can't give them extra units under a uniform margin. Another thought is that token-level errors in layers 0 through 4 only matter insofar as they propagate to the layer 5 CLS output through attention, so the real question is how much they affect that.

I'm weighing whether tokens matter unequally, since CLS attention in later layers probably focuses on some positions more than others. The CLS token's large constant dimension may also make its errors less important. I'm starting to think about how the error propagates through the layer norm, which scales each token's error by its gain and its standard deviation.

I'm thinking about whether the per-token scale of the layer-norm input matters for the error weighting. Since the error after normalization scales with each token's standard deviation, weighting the regression by the inverse squared spread could help. The per-output-dimension weights wouldn't change the ridge solution itself, but they might change which units get selected. I want to check how much the token standard deviations actually vary before deciding whether this is worth pursuing.

I'm now testing whether the new data gives better agreement at m=1536, which would beat the old m=2048 point. I'm also considering how the weighted regression, with per-column and per-token weights from the layer-norm gain, could shift which units get picked.

I'm settling on a noise target of around σ≈0.25–0.3 for the margin, which looks achievable at m1536 with the new data. Layer dropping doesn't help here, since the 5-layer version's agreement of 0.955 already costs more than the m1536 error.

I'm considering a corrective fit for the token layers: re-probing the edited model on the original 32 ids to get aligned token-level activations, then fitting toward the base outputs with ridge shrinkage. The L5 test already showed no gain at the CLS level, so I doubt this helps much at the token level either.

I'm moving on from the corrective fit idea, since it likely won't help at the token level either. Next I'll rebuild with the new data at m=1536, and maybe also test m=1280 or m=1792 to find the best size, with one pass per candidate.



Bias shifts are too small to matter, and the flips mostly drop "strategy" predictions that were already slightly underpredicted; I'm testing a 1/σ² token weighting on held-out post-LN error next.



## tool_use: run_python

```json
{
 "code": "exec(open('seq.py').read())\ndef evalw(L, m, lam_rel, weighted, gsel):\n    A, S = all_data(L); te = (S<100)&(S%2==1); tr=~te\n    pre = f\"roberta.encoder.layer.{L}.\"\n    W1 = base(pre+\"intermediate.dense.weight\").astype(np.float64); b1 = base(pre+\"intermediate.dense.bias\")\n    W2 = base(pre+\"output.dense.weight\").astype(np.float64); b2 = base(pre+\"output.dense.bias\")\n    g2 = base(pre+\"output.LayerNorm.weight\").astype(np.float64)\n    A = A.astype(np.float64); H = gelu(A@W1.T+b1); Z = A + H@W2.T + b2\n    sig = Z.std(1)\n    w = 1/sig**2 if weighted else np.ones(len(A))\n    colw = (g2[:,None]**2*W2**2).sum(0) if gsel else (W2**2).sum(0)\n    Htr = H[tr]; wt = w[tr]/w[tr].mean()\n    mu = (wt[:,None]*Htr).sum(0)/wt.sum()\n    sc = ((wt[:,None]*(Htr-mu)**2).sum(0)/wt.sum())*colw\n    keep = np.sort(np.argsort(-sc)[:m]); drop = np.setdiff1d(np.arange(3072), keep)\n    T = H[:,drop]@W2[:,drop].T\n    Hk = H[:,keep]; mk = (wt[:,None]*Hk[tr]).sum(0)/wt.sum(); mt = (wt[:,None]*T[tr]).sum(0)/wt.sum()\n    X = (Hk[tr]-mk)*np.sqrt(wt)[:,None]; Tt = (T[tr]-mt)*np.sqrt(wt)[:,None]\n    G = X.T@X; B = np.linalg.solve(G+lam_rel*np.trace(G)/m*np.eye(m), X.T@Tt)\n    err = T[te] - ((Hk[te]-mk)@B + mt)\n    errc = err - err.mean(1,keepdims=True)\n    post = (g2*errc)/sig[te][:,None]\n    # reference: varying part of post-LN output\n    return np.sqrt((post**2).sum(1)).mean()\nfor L in [0,1,3,4]:\n    r = [evalw(L,1024,0.1,wgt,gs) for wgt,gs in [(False,False),(True,False),(False,True),(True,True)]]\n    print(L, np.round(r,4))"
}
```

