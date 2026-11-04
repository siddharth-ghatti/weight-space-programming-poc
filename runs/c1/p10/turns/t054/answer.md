## reasoning summary

I've confirmed the formula matches to about 1e-7 relative error. With W1 and b1 zeroed, GELU(0) gives zero, so the FFN contributes nothing. That means for m=3072 I can compute the final CLS features for all 1496 samples straight from the base model's stored mean and CLS hidden states at layer 9, since layers 0 through 8 stay unchanged if I drop the bias edits.

I'm checking whether base layers 0 through 8 match this structure's layers, which they do since outputs there depend only on those layers and the token set is the same. That makes the m=3072 features computable from base reads. Since that frees passes, I'm weighing a smaller m=2816 experiment: read layers 9 and 10 for the first 1024 ids, and only read the remaining 472 if the drop is at most 0.15, otherwise fall back to m=3072.

Next I'm weighing whether to improve the compensation before running the m=2816 experiment, since the 11% mean shift at layer 1 hints that the bias correction was off. I'd like to check the predicted post-LN mean shift for layer 0, but I can't do that without the per-token FFN outputs. I'm also considering testing m=2816 on the 1024-id subset now and extending to m=2688 later if it holds up.

I'm estimating how much the observed layer-1 mean shift should correct layer 0's output bias, using a rough pre-LN scale to convert the post-LN difference back. The trouble is that deeper layers compound the shift and a single read can't separate each layer's contribution, so this correction only addresses the first layer. The kernel head gets refit on the edited features anyway, which partly absorbs any constant offset.

I'm weighing whether the mean shift matters, since the kept units and the compensation for pruned ones live in structure.json and I want to check how the compensation was computed before running m=2816.

I'm checking whether the edits for m=2560 match the formula, using the proxy means from ffn_proxy, so I can reconstruct the compensation method.



The formula matches the pooling output to 1e-7, so unpruned layers 0–8 can give all 1496 final [CLS] features with no extra passes, and the m=3072 fallback needs one verification pass. Next I'll rebuild the pruning edits (top-m by importance, b2 += W2_pruned·mean) so I can regenerate them for other values of m.



## tool_use: run_python

```json
{
 "code": "p=np.load('ffn_proxy.npz'); s=structure()\nfor L in [0,4,8]:\n    keep=np.array(s['ffn_keep'][str(L)] if isinstance(s['ffn_keep'],dict) and str(L) in s['ffn_keep'] else s['ffn_keep'][L] if isinstance(s['ffn_keep'],dict) else s['ffn_keep'])\n    imp=p[f'imp{L}']; top=np.sort(np.argsort(-imp)[:2560])\n    print(L, len(keep), np.array_equal(np.sort(keep), top))\n    pr=np.setdiff1d(np.arange(3072), keep)\n    W2=base(f'bert.encoder.layer.{L}.output.dense.weight'); b2b=base(f'bert.encoder.layer.{L}.output.dense.bias')\n    pred=b2b + W2[:,pr]@p[f'mean{L}'][pr]\n    cur=W(f'bert.encoder.layer.{L}.output.dense.bias')\n    print('  comp match err', np.abs(pred-cur).max(), 'changed on RES?', np.abs(cur-b2b)[RES].max())\nprint(type(s['ffn_keep']), list(s['ffn_keep'].keys())[:3] if isinstance(s['ffn_keep'],dict) else '')\n"
}
```

