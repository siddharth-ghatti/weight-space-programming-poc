## reasoning summary

I'm getting noise levels around 0.16 at m=2048 and agreement at 98.4%, which is still not good enough. The noise drops as m grows, so I'd expect roughly 0.08 at m=2560. The CLS error after layer 4 looks large relative to how small its varying part is, so I need a better method.

I'm weighing which layers' pruning drives most of the final error, and whether reading token-level states to trace error accumulation is worth the cost. A sequential refit looks more promising: for each layer in turn, I'd recalibrate against base targets using the already-edited earlier layers, since the post-LN output normalizes the error in the FFN.

I'm now setting the refit objective: keep the base-side targets from the saved probe data, choose which FFN units to keep using the edited-model inputs, and solve a ridge-regularized least-squares problem that pulls the new down-projection weights toward the original ones.

I'm weighing whether to add more probe data with longer paragraphs, since that would give the refit more tokens. It would also mean running base probes on the new text, so the cost roughly doubles. The sequential refit would need about five probe passes, and the gain from it may be modest.

I'm weighing whether to pull more tokens per probe pass with longer paragraphs, since token_hidden only returns 32 ids and per-pass counts run to a few thousand tokens. Short paragraphs would make the probe set noisier, so the trade-off is representativeness against token count. Choosing units by a downstream-importance proxy looks too complicated, so I'm leaning toward accepting a simpler approach.

Checking whether the noise has a systematic bias: I'll compare the mean logit differences per class, since a constant shift could be fixed with a bias correction in out_proj.bias rather than by changing the pruning method.

I'm planning three per-layer-group pruning runs (layer 5 alone, layers 0-2, and layers 3-4) to see which layers drive most of the noise. If one group dominates, I can focus compensation there, and I already have partial numbers at m=1024 and m=1536 to compare against.

I'm checking how much of the layer-5 error comes through to the CLS output. Its FFN pruning error is about 0.077 relative to the centered residual, and the CLS varying part at layer 6 is fairly large. I can measure the downstream change directly: take the exact base layer-5 CLS FFN inputs for all 1000 samples, compute the residual change for each, and see how it propagates through the post-LN output.

I'm going to measure the layer-5 error relative to the spread of the CLS representation across examples, rather than running the model. I'll fit the compensation on the even-indexed samples, then check the ratio of the error norm to the varying part on the odd ones at m=2048.



At m=2048 the agreement is 98.4% with margin noise 0.158, still above my comfort level and flattening, so I'm estimating offline how much of the remaining error comes from layer 5's own FFN pruning, using its pre-LN error on the `<s>` token on the held-out odd paragraphs.



## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'builder.py').read())\nA = ffn_data(5); pre=\"roberta.encoder.layer.5.\"\nW1 = base(pre+\"intermediate.dense.weight\"); b1 = base(pre+\"intermediate.dense.bias\"); W2 = base(pre+\"output.dense.weight\"); b2=base(pre+\"output.dense.bias\")\nH = gelu(A@W1.T+b1); R = A + H@W2.T + b2\neven = np.arange(1000)%2==0; odd=~even\nRv = R[odd]-R[even].mean(0)\nprint('varying norm of pre-LN R at L5 CLS', np.linalg.norm(Rv,axis=1).mean(), 'full', np.linalg.norm(R[odd]-R[odd].mean(1,keepdims=True),axis=1).mean())\nfor m in [1024,1536,2048,2560]:\n    sc = H[even].var(0)*(W2**2).sum(0); keep=np.argsort(-sc)[:m]; drop=np.setdiff1d(np.arange(3072),keep)\n    Hk=H[:,keep]; mk=Hk[even].mean(0); T=H[:,drop]@W2[:,drop].T; mt=T[even].mean(0)\n    Xk=Hk[even]-mk; G=Xk.T@Xk\n    out=[]\n    for lr in [None,0.01,0.1,1.0]:\n        if lr is None: B=np.zeros((m,768))\n        else: B=np.linalg.solve(G+lr*np.trace(G)/m*np.eye(m), Xk.T@(T[even]-mt))\n        D=T[odd]-mt-(Hk[odd]-mk)@B; D-=D.mean(1,keepdims=True)\n        out.append(f\"lam{lr}: {np.linalg.norm(D)/np.linalg.norm(Rv):.3f}\")\n    print(m, ' '.join(out))"
}
```

