## reasoning summary

The compensation vectors have a real magnitude, so the pruned units clearly contribute something meaningful to the mean output. The two proxies disagree by a lot, which makes me doubt how well either one tracks the true contribution. I can't measure the true FFN output directly, so I'm checking whether the FFN input can be recovered from what's exposed, like hidden states, attention probabilities, pooled outputs, and logits.

I'm checking whether I can set up edited models so that the layer output reveals the normalized attention-side input, but zeroing the FFN and resetting the second layer norm only gives me a normalized version of that input, not the raw values. The mean and scale get lost in that normalization, so I'm considering whether freezing the first layer norm to an identity transform would recover it, though that changes the very input I want to measure.

I'm working out whether the attention-side input can be recovered exactly. Normalizing removes the per-token mean and scale, but those two unknowns can be solved from the constraint that the inverted LayerNorm output has zero mean and unit variance across dimensions, so the input is recoverable. Still, that means one edited pass per layer, six in total, which seems costly, and I'm weighing whether that's worth it.

Alternatively, I could get the true FFN input for every layer at once, but the intermediate layers' FFNs can't be zeroed without changing what downstream layers see. Maybe a smarter trick works, though. With the true FFN inputs, I could compute the exact unit activations, which would settle the proxy question, but I'd still need to know whether that's worth the cost.

I'm weighing whether I can get the true FFN inputs without reimplementing the forward pass in numpy, since the sandbox rules forbid that. Computing per-unit activations from those inputs would count as forward computation, so I'd rather stick to Gaussian statistics of the inputs instead of exact per-token activations.

I'm weighing whether a direct read of the FFN activation statistics is possible through an engineered output, since the final output has only 768 dimensions against the FFN's 3072 units and the residual can't be cleanly removed. Rather than keep chasing that, the simpler route looks like testing the proxies empirically against the held-out metric.

I'm weighing two options for the pruning experiments: pruning FFN units across all layers, or dropping whole layers. I'll run a first pass with the averaged proxies for FFN pruning at 1536 units to see how much damage it causes, then test layer dropping with a pass of its own. For layer dropping I still need to work out how the next layer's input should be compensated.

I'm working out how to compensate when dropping a layer: a diagonal affine map can be folded into the previous layer's second LayerNorm, while a full linear map could go into the next layer's attention projections instead.

Now I'm checking whether a full linear map fits the residual stream better than the diagonal version, using ridge regression on split halves of the tokens to avoid overfitting.



The compensation vectors are large and the two proxies disagree, so the FFN-input proxy is noisy, especially in layers 0 and 5. Next I'll compare identity, diagonal-affine and full-linear layer-drop compensation on a held-out token split before spending more passes.



## tool_use: run_python

```json
{
 "code": "mask=np.load('pass_2/attention_mask.npy').astype(bool)\nHs=[np.load(f'pass_2/token_hidden_L{l}.npy') for l in range(7)]\nn=mask.shape[0]; trs=np.arange(n)<24\nfor t in range(1,6):\n    a=Hs[t][mask & trs[:,None]].astype(np.float64); b=Hs[t+1][mask & trs[:,None]].astype(np.float64)\n    at=Hs[t][mask & ~trs[:,None]].astype(np.float64); bt=Hs[t+1][mask & ~trs[:,None]].astype(np.float64)\n    ma,mb=a.mean(0),b.mean(0)\n    den=((bt-bt.mean(0))**2).sum()\n    out=[]\n    out.append(1-((bt-at)**2).sum()/den)\n    slope=((a-ma)*(b-mb)).mean(0)/(a.var(0)+1e-9)\n    out.append(1-((bt-((at-ma)*slope+mb))**2).sum()/den)\n    for lam in [1,10,100]:\n        A=np.linalg.solve((a-ma).T@(a-ma)+lam*np.eye(768),(a-ma).T@(b-mb))\n        out.append(1-((bt-((at-ma)@A+mb))**2).sum()/den)\n    print(f'drop layer {t}: heldout R2 identity {out[0]:.3f} diag {out[1]:.3f} full(l=1,10,100) {np.round(out[2:],3)}')"
}
```

