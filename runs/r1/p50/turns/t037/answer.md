## reasoning summary

I'm checking how much the layer 5 FFN pruning adds at m=2048: about 0.063 against a total final error of 0.151. Taking that out in quadrature leaves roughly 0.137 coming from the earlier layers, so layers 0 through 4 account for most of the error. Layer 5 still matters, though, with 0.104 at m=1536 and 0.149 at m=1024.

The upstream error at m=1024 is 0.319 in final centered relative error, with layer 5 at full rank, so layers 0 through 4 are the main source. Their token-level errors reach the output through layer 5's attention. I'm weighing whether the earlier layers matter more, since their errors pass through more layers, or the later ones, since they sit closer to the output.

I'm weighing whether a sequential refit that corrects upstream errors could help much, and I'm doubtful it would double the gain. I'd also like to see if the in-sample least-squares bound at m=2048 is well below the held-out error, since that would suggest more calibration tokens could help.

I'm wondering whether I could recover the layer inputs from the base model's outputs instead of running the forward pass again, since the residual stream constrains them. But solving a + f(a) = R would mean iterating the FFN in numpy, which is close to reimplementing the forward pass, so I'm not going to pursue that. Reading token states from several layers in a single base pass looks like the cheaper route.

I'm checking whether data is the bottleneck, so I'll compare the in-sample and held-out error gap at m=2048 for layers 0 through 4 before deciding whether more calibration tokens are worth collecting.



Layer 5's own pruning accounts for about 0.063 of the 0.151 centered error, so roughly 0.137 sits in upstream layers 0-4 and their token-level errors dominate. I'll compare held-out and in-sample least-squares error at m=2048 and 1536 to see whether more probe tokens would help.



## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'builder.py').read())\nM = np.load(WORK/'pass_4/attention_mask.npy').astype(bool)\npara = np.repeat(np.arange(32)[:,None], M.shape[1], 1)[M]; tr = para%2==0; te=~tr\nfor Lr in range(5):\n    pre=f\"roberta.encoder.layer.{Lr}.\"\n    W1 = base(pre+\"intermediate.dense.weight\"); b1 = base(pre+\"intermediate.dense.bias\"); W2 = base(pre+\"output.dense.weight\"); b2=base(pre+\"output.dense.bias\")\n    A = ffn_data(Lr); H = gelu(A@W1.T+b1); R = A+H@W2.T+b2\n    Rv = R[te]-R[tr].mean(0)   # token-varying part\n    out=[]\n    for m in [1536,2048]:\n        sc=H[tr].var(0)*(W2**2).sum(0); keep=np.argsort(-sc)[:m]; drop=np.setdiff1d(np.arange(3072),keep)\n        Hk=H[:,keep]; T=H[:,drop]@W2[:,drop].T\n        mk=Hk[tr].mean(0); mt=T[tr].mean(0); Xk=Hk[tr]-mk; G=Xk.T@Xk\n        for lr in [None,0.3,1.0,3.0]:\n            B=np.zeros((m,768)) if lr is None else np.linalg.solve(G+lr*np.trace(G)/m*np.eye(m), Xk.T@(T[tr]-mt))\n            D=T[te]-mt-(Hk[te]-mk)@B\n            out.append(f\"m{m} lam{lr}: {np.linalg.norm(D)/np.linalg.norm(Rv):.3f}\")\n        # in-sample bound on test half\n        X2=Hk[te]-Hk[te].mean(0); T2=T[te]-T[te].mean(0); B2=np.linalg.lstsq(X2,T2,rcond=None)[0]\n        out.append(f\"m{m} insample: {np.linalg.norm(T2-X2@B2)/np.linalg.norm(Rv):.3f}\")\n    print(f\"L{Lr}\", ' | '.join(out))"
}
```

