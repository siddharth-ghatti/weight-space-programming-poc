## reasoning summary

Comparing errors against the token-varying part, the layers 0-2 show larger errors (around 0.15-0.19 at m=2048) than layers 3-4. The in-sample bound looks much lower, but with 2500 tokens against 2048 regressors that is close to interpolation, so I don't trust it. Held-out least squares barely beats the mean-compensation baseline, which points to a data shortage; I'm thinking about how much a tenfold increase in data would change that.

I'm noting that with only half the tokens used for training, the held-out fit has room to improve once all 5049 tokens are available. The layers 0-2 are the bottleneck, since early FFN layers spread their work across many units for token-specific features. Next I want to run a learning curve at L1 with m=2048, training on 1/4, 1/2 and 3/4 of the paragraphs to estimate how much more data would help.



Layers 0-2 are the main error source, at 0.15-0.19 at m=2048, while layers 3-4 are small; the LS compensation also has only about 2,500 training tokens for 2,048 regressors. I'm running a quick learning curve on layer 1 to see whether more probe tokens would help.



## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'builder.py').read())\nM = np.load(WORK/'pass_4/attention_mask.npy').astype(bool)\npara = np.repeat(np.arange(32)[:,None], M.shape[1], 1)[M]\nte = para>=24\nfor Lr in [1,2]:\n    pre=f\"roberta.encoder.layer.{Lr}.\"\n    W1 = base(pre+\"intermediate.dense.weight\"); b1 = base(pre+\"intermediate.dense.bias\"); W2 = base(pre+\"output.dense.weight\"); b2=base(pre+\"output.dense.bias\")\n    A = ffn_data(Lr); H = gelu(A@W1.T+b1); R = A+H@W2.T+b2\n    m=2048\n    for ntr in [8,16,24]:\n        tr = para<ntr\n        Rv = R[te]-R[tr].mean(0)\n        sc=H[tr].var(0)*(W2**2).sum(0); keep=np.argsort(-sc)[:m]; drop=np.setdiff1d(np.arange(3072),keep)\n        Hk=H[:,keep]; T=H[:,drop]@W2[:,drop].T; mk=Hk[tr].mean(0); mt=T[tr].mean(0); Xk=Hk[tr]-mk; G=Xk.T@Xk\n        res=[]\n        for lr in [None,1.0]:\n            B=np.zeros((m,768)) if lr is None else np.linalg.solve(G+lr*np.trace(G)/m*np.eye(m), Xk.T@(T[tr]-mt))\n            D=T[te]-mt-(Hk[te]-mk)@B; res.append(np.linalg.norm(D)/np.linalg.norm(Rv))\n        print(f\"L{Lr} ntr_paras={ntr} tokens={tr.sum()} meancomp {res[0]:.3f} LS {res[1]:.3f}\")"
}
```

