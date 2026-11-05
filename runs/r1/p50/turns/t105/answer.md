## reasoning summary

I'm weighing why short sequences could matter here: they carry more noise since there are fewer tokens to average over. The training data leans toward the longest sequences, and the fitting set is built from 32 probe ids plus the 64 longest ids. I'm also considering whether per-token weights inversely proportional to sequence length might rebalance things.

I'm working out why the noise scales the way it does: if each token's error averages into the CLS output, the per-sequence logit noise should fall roughly as 1/sqrt(length), and the observed noise by length bucket fits that closely. That suggests fitting weights of about 1/len², which would lean heavily toward short sequences, and I'm checking whether that conflicts with the goal of the fit.

I'm checking whether the training data has enough short sequences to matter, since the token-level error may differ for them: they only cover early positions, while the long sequences dominate. I'll look at the lengths across all the data to see how many short ones there actually are.

I'm now weighing whether to rebalance the fit toward short sequences, since the per-sequence noise appears to scale like 1/sqrt(length). Weighting by 1/len² would help offset the long-sequence bias in the data, but there are few short sequences, so the effective sample size would be small. Collecting a batch of the shortest sequences seems like the better route, given the remaining passes and the cost of sequential probing.

I'm planning to pull the per-token holdout error against position and sequence length from the existing data, to see whether short sequences really err more per token or whether the 1/sqrt(length) pattern is just averaging.

I'm checking whether per-token error changes with position or with overall sequence length, since short sequences are mostly early-position tokens that also show up in longer ones. If the error is about the same regardless of length, the noise difference is just averaging and there's no targeted fix. I'll pull the holdout error split by position bucket and by sequence length.



Paragraphs under 60 tokens show logit error about 1.8 times that of 130–200-token paragraphs, which fits independent per-token error that matters most in short paragraphs. Since my compensation data is dominated by the longest paragraphs, I'm checking whether per-token holdout error depends on position or length in the existing data.



## tool_use: run_python

```json
{
 "code": "exec(open('seq.py').read())\nrng=np.random.default_rng(4)\nfor L in [0,1,2,4]:\n    A,S=all_data(L); A=A.astype(np.float64)\n    starts=np.r_[True,S[1:]!=S[:-1]]; pos=np.zeros(len(S),int)\n    seqid=np.cumsum(starts)-1; first=np.where(starts)[0]; pos=np.arange(len(S))-first[seqid]\n    slen=np.bincount(seqid)[seqid]\n    useq=np.unique(S); te_seq=rng.choice(useq,len(useq)//4,replace=False); te=np.isin(S,te_seq); tr=~te\n    pre=f\"roberta.encoder.layer.{L}.\"\n    W1=base(pre+\"intermediate.dense.weight\").astype(np.float64); b1=base(pre+\"intermediate.dense.bias\")\n    W2=base(pre+\"output.dense.weight\").astype(np.float64); b2=base(pre+\"output.dense.bias\"); g2=base(pre+\"output.LayerNorm.weight\").astype(np.float64)\n    H=gelu(A@W1.T+b1); U=A+H@W2.T+b2; sig=U.std(1,keepdims=True)\n    m=1536; sc=H[tr].var(0)*(W2**2).sum(0); order=np.argsort(-sc); keep=np.sort(order[:m]); drop=order[m:]\n    T=H[:,drop]@W2[:,drop].T; X=H[:,keep]; mu=X[tr].mean(0); Xc=X[tr]-mu; G=Xc.T@Xc\n    B=np.linalg.solve(G+0.1*np.trace(G)/m*np.eye(m), Xc.T@(T[tr]-T[tr].mean(0)))\n    E=T-((X-mu)@B+T[tr].mean(0)); en=(((E-E.mean(1,keepdims=True))/sig*g2)**2).sum(1)\n    out=[]\n    for lo,hi in [(1,10),(10,30),(30,60),(60,120),(120,600)]:\n        s=te&(pos>=lo)&(pos<hi); out.append(f'pos{lo}-{hi}:{en[s].mean():.2f}')\n    s_last=te&(pos==slen-1); out.append(f'</s>:{en[s_last].mean():.2f}')\n    print(L,out, 'seq lens in data: min %d median %d'%(np.bincount(seqid).min(), np.median(np.bincount(seqid))), flush=True)\n"
}
```

