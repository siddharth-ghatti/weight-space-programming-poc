## reasoning summary

The practice numbers look off next to the flip counts: at margins 0 to 0.5 the top-1 hit rate is about 0.62, yet among the 20 flips the baseline was right 18 times and the edited model only 2. That's far more lopsided than chance would give, so the flips seem to go against the gold label, which would mean the pruning is stripping out information tied to specific training items rather than just adding random noise.

Pruned FFN units may hold memorized training labels, so removing them shifts predictions on training items back toward the generalized answer. On test items that memorization is absent, so flips there should be more balanced and do less harm than the in-sample numbers suggest. The real remaining worry is variance across the test flips, which is what matters for the chance of a larger loss.

I'm estimating the test-set variance here: with around 20 flips the chance of a loss of 5 or more is roughly 20%, so the variance is still large even if the mean loss is small. I'm also second-guessing my estimate of how many flips will appear on test, which is uncertain. Reducing that count would need either a larger budget or better data weighting for short sequences, which might cut flips by 10-15%.

I'm weighing whether a more aggressive model size is worth the risk, since the scoring might reward it. I'm aiming for roughly a 10% chance of failing, and at m1536 the test flip count gives somewhere around 6 to 20% risk. Better data and weighting for the fidelity would tighten that, and m1792 looks like it would bring the risk down to about 4 to 13%, for about 2.4M more parameters.

I'm considering whether a bit more data from medium-length sequences, weighted by length, would tighten fidelity at m1536 before deciding on model size. The extra collection would add roughly 10% more tokens over five passes, so the gain might be modest, and I'd still want to check whether it's worth it.

I'm considering whether a sequential probe could pull FFN inputs for several layers in one pass, but each probe at layer L replaces the input that layer L+1 would need, so that doesn't work. I'm also checking whether attention outputs could expose what I need instead.

I'm considering cheaper data options: pulling CLS-position inputs for about 1000 ids avoids the token-level cost, though CLS errors are already tiny. Since collecting more data is expensive, I'll first try reweighting per-token by 1/length, which costs nothing and might match the objective better.

I'm checking whether per-token weights of 1/len² are the right fix: that's roughly what the objective implies if error per token doesn't depend on sequence length. The catch is that it would concentrate weight on the shortest training sequences, which shrinks the effective sample size without clearly matching the test distribution better.

I'm weighing whether position-dependent weighting could help, but estimating it, early-position error is only about 25% higher and covers roughly a fifth of a 50-token sequence, so the gain looks like about 5% at most. That's small enough that it probably isn't the lever I need.

I'm wondering whether greedy forward selection of FFN units, using a Schur-complement update with ridge, could beat the earlier OBS result now that there's about 21k tokens of data instead of a few thousand. My plan is to test it on holdout for layer 1 at m1536.

I'm estimating the computational cost of greedy forward selection: a 3072×3072 covariance and about 1536 steps of rank-one updates, which should take a minute or two in numpy. Then I'll implement the objective, minimizing the trace of the residual covariance after regressing on the chosen units, with ridge damping for stability, and the gain for each candidate from the projected W2 column.

I'm about to compute the greedy selection step: pick the unit with the largest gain, then update the covariance and the W2-projected matrix via rank-one updates. That's about 18 billion multiply-adds across roughly 1536 steps, which I estimate at 30 to 60 seconds in numpy.

I'm going to run the greedy selection with the ridge-compensation fit and compare its holdout error against the variance ranking, starting with layers 1 and 4.



Base's top-2 class is right only about 36% of the time at low margins, but the edited model's 20 flips went against gold 18 to 2, which suggests pruned units held item-specific memorisation, so test flips should be roughly symmetric. I'm next re-testing greedy compensation-aware selection against variance ranking on holdout, since there's now 4× more data.



## tool_use: run_python

```json
{
 "code": "exec(open('seq.py').read())\nimport time\nrng=np.random.default_rng(5)\ndef greedy_select(Htr, W2, m, delta=0.05):\n    Hc=Htr-Htr.mean(0); S=(Hc.T@Hc)/len(Hc); S+=delta*np.diag(np.diag(S))\n    W2S=W2@S; chosen=[]; alive=np.ones(S.shape[0],bool)\n    for it in range(m):\n        d=np.diag(S).copy(); g=(W2S**2).sum(0)/np.maximum(d,1e-12); g[~alive]=-1\n        j=int(np.argmax(g)); chosen.append(j); alive[j]=False\n        s=S[:,j].copy(); sj=s[j]\n        S-=np.outer(s,s)/sj; W2S-=np.outer(W2S[:,j],s)/sj\n    return np.sort(chosen)\nfor L in [1,4]:\n    A,S_=all_data(L); A=A.astype(np.float64)\n    useq=np.unique(S_); te_seq=rng.choice(useq,len(useq)//4,replace=False); te=np.isin(S_,te_seq); tr=~te\n    pre=f\"roberta.encoder.layer.{L}.\"\n    W1=base(pre+\"intermediate.dense.weight\").astype(np.float64); b1=base(pre+\"intermediate.dense.bias\")\n    W2=base(pre+\"output.dense.weight\").astype(np.float64); b2=base(pre+\"output.dense.bias\"); g2=base(pre+\"output.LayerNorm.weight\").astype(np.float64)\n    H=gelu(A@W1.T+b1); U=A+H@W2.T+b2; sig=U.std(1,keepdims=True)\n    m=1536\n    t=time.time(); kg=greedy_select(H[tr],W2,m); tg=time.time()-t\n    kv=np.sort(np.argsort(-(H[tr].var(0)*(W2**2).sum(0)))[:m])\n    for name,keep in [('var',kv),('greedy',kg)]:\n        drop=np.setdiff1d(np.arange(3072),keep)\n        T=H[:,drop]@W2[:,drop].T; X=H[:,keep]; mu=X[tr].mean(0); Xc=X[tr]-mu; G=Xc.T@Xc\n        B=np.linalg.solve(G+0.1*np.trace(G)/m*np.eye(m), Xc.T@(T[tr]-T[tr].mean(0)))\n        E=T[te]-((X[te]-mu)@B+T[tr].mean(0)); en=(((E-E.mean(1,keepdims=True))/sig[te]*g2)**2).sum(1).mean()\n        Etr=T[tr]-((X[tr]-mu)@B+T[tr].mean(0)); entr=(((Etr-Etr.mean(1,keepdims=True))/sig[tr]*g2)**2).sum(1).mean()\n        print(L,name,'test %.3f train %.3f'%(en,entr), 'overlap', len(np.intersect1d(kv,keep)), 'time %.0f'%tg, flush=True)\n"
}
```

