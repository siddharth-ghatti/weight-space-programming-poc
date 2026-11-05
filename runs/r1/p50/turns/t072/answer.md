## reasoning summary

I'm weighing 1/σ² weighting against plain token weighting, and it looks like a small gain at the deeper layers and a slight loss at L0. Selecting g2 doesn't seem to matter much. I'm now checking the scale of the post-LN error norm against the LN output norm.

I'm estimating the scale of the LN output, which seems to land around 20-25. Looking for a bigger lever, I'm considering whether a linear map of the attention output could be added to the FFN output, since near-linear dropped units could in principle be replaced that way, though the architecture doesn't offer a direct path for it.

I'm working out whether the attention output could be folded into the FFN path, but the residual only carries `a` through unchanged, so there's no direct way to add a linear map of it. Ridge regression already captures whatever linear part the kept units span, and I'm considering whether one kept unit could be repurposed with a large bias so it sits in GELU's linear regime and acts as a linear feature.

I'm considering whether the nearly linear dropped units could be captured by repurposing a few kept units with a large bias, so they act as linear projections. I'd test this by regressing the target on the kept activations plus the attention output and seeing how much the error drops. I'm also wondering whether merging kept units with highly correlated activations could free up capacity.

I'm noting that a linear map could be carried by merging units with correlated activations, which the ridge step already covers within the kept span. Next I'll test adding the attention output as extra regression features at m=1024, to see how much the error drops before trying a low-rank implementation.



Token weighting by 1/σ² and g2-weighted selection give little gain, so I'm setting them aside. I'll test whether dropped units are mostly linear in the FFN input by regressing their output on the kept units plus the input.



## tool_use: run_python

```json
{
 "code": "exec(open('seq.py').read())\ndef evallin(L, m, lam_rel, r_list):\n    A, S = all_data(L); te = (S<100)&(S%2==1); tr=~te\n    pre = f\"roberta.encoder.layer.{L}.\"\n    W1 = base(pre+\"intermediate.dense.weight\").astype(np.float64); b1 = base(pre+\"intermediate.dense.bias\")\n    W2 = base(pre+\"output.dense.weight\").astype(np.float64); b2 = base(pre+\"output.dense.bias\")\n    A = A.astype(np.float64); H = gelu(A@W1.T+b1); Z = A + H@W2.T + b2\n    den = np.sqrt(((Z[te]-Z[te].mean(0))**2).sum(1)).mean()\n    out = {}\n    for r in r_list:\n        mm = m - r\n        sc = H[tr].var(0)*(W2**2).sum(0)\n        keep = np.sort(np.argsort(-sc)[:mm]); drop = np.setdiff1d(np.arange(3072), keep)\n        T = H[:,drop]@W2[:,drop].T\n        if r>0:\n            # linear features: top-r principal directions of a that best predict residual after H_k -> use reduced-rank: first fit with full a, then SVD\n            X = np.c_[H[:,keep], A]\n        else: X = H[:,keep]\n        mx = X[tr].mean(0); mt = T[tr].mean(0); Xc = X[tr]-mx\n        G = Xc.T@Xc; d = np.diag(G).copy()\n        lam = lam_rel*np.trace(G[:mm,:mm])/mm\n        B = np.linalg.solve(G+lam*np.eye(X.shape[1]), Xc.T@(T[tr]-mt))\n        if r>0 and r<768:\n            # reduced rank on linear block: M = B[mm:] (768x768); keep rank r approx in terms of predictions\n            M = B[mm:]; U,s,Vt = np.linalg.svd((A[tr]-A[tr].mean(0))@M, full_matrices=False)\n            # project M to top-r output directions\n            P = Vt[:r].T@Vt[:r]; Mr = M@P\n            # refit H block given Mr fixed\n            R = T[tr]-mt - (A[tr]-A[tr].mean(0))@Mr\n            Hc = Xc[:,:mm]; Gh = Hc.T@Hc\n            Bh = np.linalg.solve(Gh+lam*np.eye(mm), Hc.T@R)\n            B = np.r_[Bh, Mr]\n        pred = (X[te]-mx)@B + mt\n        out[r] = round(np.sqrt(((T[te]-pred)**2).sum(1)).mean()/den,4)\n    return out\nfor L in [1,3]:\n    print(L, evallin(L, 1024, 0.1, [0, 32, 64, 128, 768]))"
}
```

