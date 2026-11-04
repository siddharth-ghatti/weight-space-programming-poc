## reasoning summary

I'm comparing the results across the five seeds: the base M9 and m1024 runs trail the M0+M9 and M0+M8 combinations, and the m1024 gap with M0 combos looks around 0.2.

I'm checking whether the M0 carry can reach the final pool, since the kernel on M0 with M9 beats M9 alone, which suggests M9 has lost information M0 still holds. I'm weighing whether reserved dimensions at an early layer could carry M0-derived features forward, given that the earlier 8-dimension compression attempts didn't help.

I'm considering whether an M0 kernel machine could run early and write its eight class scores into reserved dimensions, to be combined with the M9 kernel scores at the end. A sum of kernels corresponds to adding two separate models' scores, so I'm going to test whether a sum kernel K0 + K9 reaches the accuracy of the product kernel.

I'm weighing whether an M0-kernel machine could sit in an early layer, with one attention head pooling mean(h0) uniformly across tokens and writing its output into reserved dimensions, but I'm worried that layer 0's FFN units are already needed for content and would be squeezed by the M0 budget.

I'm considering whether the CLS token could carry M0 information at full width, since its attention in layer 0 could be made uniform and pick up the mean of the early states. The open question is whether that route avoids costing the layer-0 FFN units their content role.

I'm weighing reserving dimensions at every position to carry the M0 projection forward, which would mean zeroing those dimensions in the later layers' weights. Sixty-four reserved dimensions would cost about 8% of the content capacity, and I can't tell how that would affect the content features.

I'm weighing whether the M0 kernel signal could reach the end through a sum-kernel decomposition, and whether the 1536-unit version with M9 alone is strong enough to skip M0 altogether, given it sits close to the 97.8 threshold.

Now I'm checking whether an M1 or M0 kernel could be combined with M9 for a richer final read, since M0+M9 and M1+M9 both score above M9 alone on the full data. I'm wondering whether the existing carry heads at layer 2 could be repurposed to send more early information forward, though the eight reserved dimensions are still the bottleneck.

I'm looking at a sum kernel, f(M1) + f(M9), where an early-layer M1 or M2 kernel machine writes its 8 scores into reserved dimensions. The open question is where that computation can live, since it needs a pooled early representation and some FFN units to evaluate the kernel. One idea is to use the existing carry head in layer 2 to pass a 64-dimensional projection of mean(h2) forward, though that would spill into content dimensions.

I'm thinking about whether the carry head could write a 64-dimensional projection of the pooled mean into a subspace of the residual stream, with the layer's FFN units reading it and then subtracting it back out, so the content dimensions stay clean.

I'm working out how the cancellation has to happen: the linear passthrough units would have to subtract the carried v before the next LayerNorm, since the v that LN has already scaled sits in the pre-LN sum. Each dimension would cost about two GELU units to get a linear term, so the cancellation alone is expensive.

I'm working out the cancellation: the carry's contribution to the normalized state is scaled by the per-token standard deviation, so a subtraction would also have to remove the matching slice of the content in those same dimensions. Restoring that content back is the sticking point, since the units can't tell the carried signal apart from it.

I'm considering whether the low-variance directions of the residual stream could serve as the reserved subspace instead of coordinate dimensions. Since those directions carry little content, zeroing them out in the projections might cost very little, and it would give the carry a place to live without occupying dedicated coordinates.

I'm considering whether a non-coordinate subspace D could replace the reserved coordinate dimensions. The trouble is that LayerNorm's per-dimension gamma doesn't preserve arbitrary subspaces, so the reads and writes would have to be annihilated in diag(gamma)·D instead of D itself.

Checking whether the eight reserved dimensions already carry usable M2 and M6 scores, since the final classifier might read them directly without any new mechanism.

I'm now second-guessing the stacking setup: the reserved-dimension scores from the carry heads were fit on all practice data, so using them directly in a final classifier would over-trust them. I'd need out-of-fold versions to avoid that. Another option is to refit the carry heads at layers 2 and 6 so they produce the eight reserved-dimension values myself.

Now I'm checking whether the carried M2 and M6 ridge scores help at all, since linear M2 is only around 95% and may add little on top of the M9 kernel. I'll test stacking them onto the kernel using nested out-of-fold scores, and if that gains roughly 0.3%, it's a cheap win through the existing carry mechanism.

I'm weighing whether a carry head at layer 0 or 1 could compute scores from M0 or M1, since those pair well with M9. The catch is that the reserved dimensions get zeroed by the layer-1 output LayerNorm, so the early layers would need to write through a different path.

I'm going to test the combined model next: the kernel scores from M9 plus a linear term from the M2 scores, with the weight on the linear term chosen by cross-validation.



Five fold-seeds confirm the m=1024 cost is about 0.2–0.35%, and M0 info adds 0.4–0.7%, so I'm testing whether stacking carried linear class scores onto an M9 kernel classifier recovers part of that gain.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0, str(WORK))\nfrom lib import *\ny=labels(); d=np.load('base_feats.npz'); N=len(y); Y=np.eye(8)[y]\ndef cn(X):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.linalg.norm(X,axis=1,keepdims=True)\ndef oof_krr(K, f, lam=0.005):\n    S=np.zeros((N,8))\n    for i in np.unique(f):\n        tr=f!=i; te=f==i\n        A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0)); S[te]=K[np.ix_(te,tr)]@A\n    return S\ndef oof_ridge(X, f, lam):\n    S=np.zeros((N,8))\n    for i in np.unique(f):\n        tr=f!=i; te=f==i; Wt,b=ridge_fit(X[tr],y[tr],lam); S[te]=X[te]@Wt+b\n    return S\nZ9=cn(d['mean_hidden_L9']); K9=np.exp(Z9@Z9.T-1)\nZs={l:cn(d[f'mean_hidden_L{l}'])*np.sqrt(767) for l in [0,1,2,6]}\nfor l in [0,2,6]:\n    for lamr in [30,100,300]:\n        accs={b:[] for b in [0,0.25,0.5,1.0]}\n        for s in range(3):\n            f=folds(y,10,s)\n            S9=oof_krr(K9,f); L=oof_ridge(Zs[l],f,lamr)\n            for b in accs: accs[b].append(((S9+b*L).argmax(1)==y).mean())\n        print('M%d lam%d'%(l,lamr), {b:round(np.mean(v),4) for b,v in accs.items()}, 'lin alone', round((L.argmax(1)==y).mean(),4))"
}
```

