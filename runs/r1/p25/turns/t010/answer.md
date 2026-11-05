## reasoning summary

I'm computing per-layer statistics, the mean and covariance over the non-padded tokens, and saving them to a compact file. Then I'll estimate FFN importance for layers 0 through 5, using the layer input and output as proxies, by working out each neuron's pre-activation mean and variance and the Gaussian expectation and variance of GELU.

I'm also weighing whether the CLS token should get its own statistics, since it may behave differently from the other positions, especially at the last layer. Next I'll check how similar consecutive layers are by comparing token cosines between each layer and the one after it, and by measuring how well a diagonal affine map fits between them.



I'm computing per-layer token means and covariances and checking how similar consecutive layers are, to see how much each layer changes the residual stream and how well a diagonal affine map could stand in for a dropped layer.



## tool_use: run_python

```json
{
 "code": "mask=np.load('pass_2/attention_mask.npy').astype(bool)\nH=[np.load(f'pass_2/token_hidden_L{l}.npy')[mask].astype(np.float64) for l in range(7)]\nprint(H[0].shape)\nmus=np.stack([h.mean(0) for h in H]); covs=np.stack([np.cov(h.T) for h in H])\nnp.save('tok_mu.npy',mus); np.save('tok_cov.npy',covs)\n# CLS rows\ncls_mask=np.zeros_like(mask); cls_mask[:,0]=True\nfor l in range(6):\n    a,b=H[l],H[l+1]\n    cos=(a*b).sum(1)/np.linalg.norm(a,axis=1)/np.linalg.norm(b,axis=1)\n    # diag affine fit per dim\n    va=a.var(0); cab=((a-a.mean(0))*(b-b.mean(0))).mean(0); slope=cab/va\n    pred=(a-a.mean(0))*slope+b.mean(0)\n    r2d=1-((b-pred)**2).sum()/((b-b.mean(0))**2).sum()\n    r2id=1-((b-a)**2).sum()/((b-b.mean(0))**2).sum()\n    print(f'layer {l}: mean cos(h_in,h_out) {cos.mean():.3f}, R2 identity {r2id:.3f}, R2 diag-affine {r2d:.3f}, norms {np.linalg.norm(a,axis=1).mean():.1f}->{np.linalg.norm(b,axis=1).mean():.1f}')"
}
```

