## reasoning summary

The importance spread is fairly wide, since keeping 1536 units only holds about 82% of the variance, so pruning the FFN will likely cost something noticeable. The two proxies agree only moderately, around 0.7 to 0.8, so the proxy itself is noisy. I'd rather use the true FFN input, and I'm checking whether the attention output can give me that.

I'm weighing whether I can get the true FFN input without reimplementing the attention sublayer, which seems off-limits. So I'll go with the proxy and run a first experiment: prune to 1536 units per layer for 6 layers, using combined importance and a bias correction, then compare the edited model's logits and last-layer CLS hidden states against the base.

I'm weighing a cheaper compensation for the pruned FFN units, one that avoids computing activations. Fitting the kept units' output weights by least squares would need joint activation statistics for every unit pair, and at 3072 units that quadrature is too heavy to be worth it.

I'm weighing whether to compute the FFN activations directly on the proxy inputs, but that would amount to running part of the model on proxies rather than the real forward pass, so I'm leaning toward avoiding it. Instead I'll stick with Gaussian statistics for the compensation, since a pairwise quadrature over all 3072 units looks too expensive to be worth it.

Now I'm thinking the sublayer compensation is just the same computation in a different form, so it's really a statistical approximation either way. I'll keep that and move on. Next I'm considering whether a refit head could recover some of the lost agreement after pruning six layers to 1536 units each.

I'm checking whether the bias compensation is accurate: the pruned units' mean activations come from the proxy, so any error there gets summed across 1536 units. I want to compare how much the summed Eg contributions differ between the input-side and output-side estimates to see how sensitive it is.



FFN pruning will likely cost accuracy, since keeping half the units retains only about 82% of output variance and the two proxies agree on just 70–80% of top units. I'm checking how consistent the bias-compensation vectors are between the two proxies first.



## tool_use: run_python

```json
{
 "code": "for l in range(6):\n    d=np.load(f'imp_L{l}.npz'); U=base(f'roberta.encoder.layer.{l}.output.dense.weight').astype(np.float64)\n    b2=base(f'roberta.encoder.layer.{l}.output.dense.bias')\n    keep=np.argsort(-d['I'])[:1536]; pr=np.setdiff1d(np.arange(3072),keep)\n    ci=U[:,pr]@d['Eg_in'][pr]; co=U[:,pr]@d['Eg_out'][pr]\n    full_i=U@d['Eg_in']; \n    print(l, 'comp norm in %.2f out %.2f diff %.2f | full mean FFN out norm %.2f | bias norm %.2f'%(np.linalg.norm(ci),np.linalg.norm(co),np.linalg.norm(ci-co),np.linalg.norm(full_i),np.linalg.norm(b2)))"
}
```

