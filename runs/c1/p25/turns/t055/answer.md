## reasoning summary

I'm noting that the M8 GELU kernel with a=1 plus the linear terms reaches about 97.65, and M9 gets to about 98.05, while a=2.5 breaks because of non-PSD issues. I could also try adding the CLS feature or tuning the scaling of the linear parts. The actual u differs from z(M8), since it's normalized from x_cls plus α times the mean, and per-dimension scaling by 1/sd from the u reads would be equivalent to z-scoring u.

I'm weighing k=9 against k=10: k=9 (M8) sits near 97.65 CV, while k=10 (M9) reaches about 98.05 and should clear the 97.8% test bar more reliably. A rough paired-disagreement estimate puts k=9 at roughly a one-in-five chance of failing the threshold, versus around 7% for k=10, and k=10 is the larger model at 95.3M parameters against 88.2M.

I'm weighing the failure risk against model size: k=10 looks safer, and a reduced-width variant at m=2560 would come in slightly under k=9's parameter count while keeping that margin. I'm checking whether that smaller width could hold up.

I'm considering whether a reduced-width model at m=2560 would lose much accuracy, since pruning about 17% of the units with bias folding could cost roughly 0.2 points, but I don't know that yet. Testing it would take a pass with the edited model, and I have enough budget left to run a few experiments, though building and verifying the result will also need passes.

I'm wondering whether the importance metric itself is the weak point: the current proxy uses the mean activation, but dropping a unit actually costs its variance times its output-weight norm, so a variance-based score might prune more safely. I'd need to check what ffn_imp.npz actually stores, and computing per-unit variance from the FFN intermediates would mean an extra numpy pass over the sampled sentences.

I'm going to test whether a reduced-width k=10 model at m=2560 keeps the accuracy of M9, using variance-based importance with mean folding. If it does, it would be smaller than k=9 at full width and still score higher. I could also try a further-reduced m=2048 variant, and I'd need the pool slot to use the same width.

Next I'll check what the saved importance file holds, and pull the 64-sentence token reads from the pass 4 output across all layers.



K=10 looks safer than k=9 for the bar (about 7% versus 20% risk of failing), and a narrower k=10 variant at m=2560 would be smaller than k=9 at full width. I'm checking what ffn_imp.npz holds, since the variance-based importance score may be the right one for that pruning.



## tool_use: run_python

```json
{
 "code": "F=np.load('ffn_imp.npz'); print(list(F.keys())[:30])\nfor k in list(F.keys())[:4]: print(k, F[k].shape, F[k][:5])\nm=json.load(open('pass_4/meta.json')); print(m.keys(), m.get('layer_map'), len(m['ids']))\nam=np.load('pass_4/attention_mask.npy'); print(am.shape, am.sum())\nprint(np.load('pass_4/token_hidden_L0.npy', mmap_mode='r').shape)"
}
```

