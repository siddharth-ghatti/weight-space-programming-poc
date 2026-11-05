## reasoning summary

Token-level errors are what dominate here, since the CLS errors are tiny and the hypothesis doesn't hold. Test error energy runs about 1.35 times the training error, which points to the problem being pure token-level approximation rather than something in the classification token.

I'm now weighing a non-destructive probe variant: zeroing the FFN output and setting LN2 to identity would make the layer's output equal to the normalized FFN input, but that changes what the later layers receive, so it doesn't work as a clean measurement.

I'm considering a non-destructive way to get a_L: from the layer output h_L I could recover the FFN sum u = a + f(a) by inverting the LN2 scaling, but solving for a would mean fixed-point iteration on the FFN and the per-token scalars, which feels like reimplementing the forward pass and might be too fragile.

I'm weighing whether to gather more data from extra probe passes, which would add roughly 38% more tokens for an expected 5-8% error reduction, a modest gain. Those passes are still needed for candidate evaluation and for the final sequential probes, so I'm considering whether a different use of them would do better.

I'm thinking about whether unit selection that accounts for compensation could cut error by 10-30%. The notes say greedy OBS didn't beat variance ranking on holdout, but I'm not sure what was actually tested, so I'm going to compare variance ranking against a leverage-style greedy selection over the weighted covariance.

I'm formalizing the compensation-aware objective: the error energy is the trace of the weighted Schur complement, and I'm considering forward greedy selection that adds the unit giving the largest drop in that energy, though a naive run over 1536 steps and 3072 candidates looks expensive and may need incremental updates.

I'm setting up the greedy selection with a rank-1 update of the Schur complement: after adding unit j, the matrix shrinks by the outer product of its column over its variance, and the score for each candidate is the squared norm of its weighted column divided by its remaining variance. The per-step cost is dominated by the 3072-by-3072 update, so I still need to check whether that is affordable across 1536 steps.

I'm turning to the OBS-style greedy selection, which probably matches what the notes called "OBS greedy." Ridge regularization might help with overfitting on the in-sample covariance, but the notes already report no holdout gain, so I'll trust that. I'll keep the plan simple: pick m, maybe gather a bit more data, refit with lam 0.1, and verify on holdout.

I'm considering whether layers 0 and 1 might act mostly as token-identity lookups, since layer 0's FFN input is largely set by the token and its local context. If so, a token-specific mapping could stand in for some FFN units, but I can't add new parameters or lookups here because the embeddings are frozen.

I'm wondering whether the practice flip rate of 1.9% at m1536 is too optimistic, since practice margins may be inflated. I'll try to estimate the test flip rate from the empirical distribution of margin changes, since there's no held-out base model to compare against.

I'm weighing whether dropping layer 5 entirely could help, but its attention is essential, so that doesn't hold up. The other option is margin-preserving head adjustment, but flips sit at low margins with symmetric noise, so that won't reduce the risk either.

I'm going to check how much more training data would help. I'll fit the layer 1 model at m1536 with lam 0.1 on 25%, 50%, 75%, and all of the training data, then see how holdout error changes and extrapolate from there.



That hypothesis is also ruled out, since the `<s>` errors are small at every layer. I'm now measuring holdout error against training tokens for L1 and L2 at m=1536 to see whether more compensation data is worth it.



## tool_use: run_python

```json
{
 "code": "exec(open('seq.py').read())\nrng=np.random.default_rng(2)\nfor L in [1,2]:\n    A,S=all_data(L); A=A.astype(np.float64)\n    useq=np.unique(S); te_seq=rng.choice(useq,len(useq)//4,replace=False); te=np.isin(S,te_seq); trall=np.where(~te)[0]\n    trseq=np.setdiff1d(useq,te_seq); rng.shuffle(trseq)\n    pre=f\"roberta.encoder.layer.{L}.\"\n    W1=base(pre+\"intermediate.dense.weight\").astype(np.float64); b1=base(pre+\"intermediate.dense.bias\")\n    W2=base(pre+\"output.dense.weight\").astype(np.float64)\n    H=gelu(A@W1.T+b1); F=H@W2.T; den=((F[te]-F[te].mean(0))**2).sum()\n    m=1536\n    for frac in [0.25,0.5,0.75,1.0]:\n        tr=np.isin(S, trseq[:int(len(trseq)*frac)])&~te\n        sc=H[tr].var(0)*(W2**2).sum(0); order=np.argsort(-sc); keep=np.sort(order[:m]); drop=order[m:]\n        T=H[:,drop]@W2[:,drop].T; X=H[:,keep]; mu=X[tr].mean(0); Xc=X[tr]-mu; G=Xc.T@Xc\n        B=np.linalg.solve(G+0.1*np.trace(G)/m*np.eye(m), Xc.T@(T[tr]-T[tr].mean(0)))\n        E=T[te]-((X[te]-mu)@B+T[tr].mean(0))\n        print(L, frac, tr.sum(), 'test rel %.4f'%((E**2).sum()/den), flush=True)\n"
}
```

