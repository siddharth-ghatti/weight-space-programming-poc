## reasoning summary

The R2 coming out strongly negative at layer 1 for the CLS token looks like a variance problem: CLS at that depth barely changes across items, so relative errors look huge against the small spread between them. Still, the deviations should be judged against the hidden-state norm, and there may be massive-activation dimensions skewing things. I'll measure relative to

I'm checking whether the per-item error relative to the base hidden-state norm, ||ce-cb|| / ||cb||, explains the 0.836 mean R2 at layer 1. An FFN error of only a few percent shouldn't produce a large hidden-state error, unless the across-item variance I'm normalizing by is tiny.

I'm weighing whether the CLS token's FFN input sits far outside the token distribution used for the importance stats, since only 32 of the 4051 ids were CLS. If CLS acts as an attention sink at early layers, a shift in its hidden state could spread to every other token, which might explain the falling mean R2. That points toward treating position 0 separately.

I'm now considering anchoring the bias compensation at the CLS input, since that single point is nearly constant across items and seems to act as an attention sink, so zeroing the error there could matter more than matching the token average.

I'm thinking about a CLS-specific unit: if CLS has a distinctive direction, such as a massive-activation dimension, a unit could be near zero on normal tokens and large only at CLS, with its output column set to the needed correction there. That would fix the error at the CLS point without shifting the others. Existing units may already do this, so I'll check whether tok_kind.npy marks CLS, EOS, and normal tokens, then measure the per-layer error at CLS rows against normal tokens using the base FFN inputs.

I'm working out how to measure the error at each token kind, starting from the base FFN and the edited FFN evaluated at the same inputs. The scale I need to compare against is the post-LN2 hidden state, meaning the residual plus the FFN output, so I'll normalize the error by the norm of that sum.



The CLS (<s>) states deviate far more than their spread across items, likely because <s> acts as an attention sink. Next I'll check per-layer FFN error separately for <s>, ordinary and </s> tokens.



## tool_use: run_python

```json
{
 "code": "import numpy as np, scipy.special as sps\ndef gelu(z): return 0.5*z*(1+sps.erf(z/np.sqrt(2)))\ntk=np.load('tok_kind.npy'); print('tok_kind', tk.shape, np.unique(tk, return_counts=True))\nst=structure()\nfor l in range(5):\n    a=np.load(f'true_a_L{l}.npy').astype(np.float64)\n    pre=f'roberta.encoder.layer.{l}.'\n    W1=base(pre+'intermediate.dense.weight'); b1=base(pre+'intermediate.dense.bias'); U=base(pre+'output.dense.weight'); b2=base(pre+'output.dense.bias')\n    f=gelu(a@W1.T+b1)@U.T+b2\n    kp=np.array(st['ffn_keep'][str(l)] if isinstance(st['ffn_keep'],dict) and str(l) in st['ffn_keep'] else st['ffn_keep'][l])\n    W1e=W(pre+'intermediate.dense.weight')[kp]; b1e=W(pre+'intermediate.dense.bias')[kp]; Ue=W(pre+'output.dense.weight')[:,kp]; b2e=W(pre+'output.dense.bias')\n    fe=gelu(a@W1e.T+b1e)@Ue.T+b2e\n    x=a+f\n    for kind in np.unique(tk):\n        s=tk==kind\n        err=np.linalg.norm(fe[s]-f[s],axis=1); nf=np.linalg.norm(f[s]-f[tk==1].mean(0),axis=1); nx=np.linalg.norm(x[s],axis=1)\n        print(f'L{l} kind{kind} n={s.sum()}: |err| {err.mean():.2f}  |f-meanf| {nf.mean():.2f} |x| {nx.mean():.1f}  rel(err/|x|) {np.mean(err/nx):.3f}')\n"
}
```

