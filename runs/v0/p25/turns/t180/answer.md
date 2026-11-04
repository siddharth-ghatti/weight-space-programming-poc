## reasoning summary

Setting lambda to 0.005 and weighting w6 at 0.5 with the rest at 0.4, since everything is in noise. Next I'll pull the exact v5 construction details, the FFN neuron placement and the W2 and b2 setup, so I can rebuild v6 the same way.

Checking the layer 11 setup next: attention is uniform, so the CLS state picks up twice the token mean before LayerNorm, and the feed-forward weights are set so the first-layer activations reduce to a scaled dot product with the centered direction. I'm still working out how the output projection and bias combine with the residual.

Now I'm working out how the feed-forward output, scaled by the large factor S and projected through the low-variance directions U, feeds into the residual before the second LayerNorm, and whether those components dominate the variance once normalized.

I'm checking whether the final argmax survives the normalization. The large S scaling makes the U components dominate the variance, and since tanh is monotonic, the ranking of the logits should hold. I still need to pull the exact v5 build details, especially the W1 and bias setup, to reuse them.

I'm checking the saved weights against the formula: the intermediate weights should be the scaled, centered Z directions, and the output weights should match S times U times A transposed, with the bias built from b0 and the mean.



All values stay within noise, so I'm keeping lam at 0.005 and setting my6's weight to 0.5. Before rebuilding, I'll check my understanding of v5 by reconstructing its layer-11 FFN weights from kernel_v5.npz.



## tool_use: run_python

```json
{
 "code": "import sys, os; sys.path.insert(0, str(WORK)); os.chdir(WORK)\nimport numpy as np\nk=np.load('kernel_v5.npz'); U=np.load('U_v2.npy'); print(U.shape)\nW1=W('bert.encoder.layer.11.intermediate.dense.weight'); b1=W('bert.encoder.layer.11.intermediate.dense.bias')\nW2=W('bert.encoder.layer.11.output.dense.weight'); b2=W('bert.encoder.layer.11.output.dense.bias')\nW1r=(1/768)*k['C']/k['sd'][None,:]; b1r=-W1r@k['mu']\nprint('W1',np.abs(W1-W1r).max(), np.abs(W1).max(), 'b1',np.abs(b1-b1r).max())\nS=100\nW2r=S*U@k['A'].T; b2r=U@(S*k['b0']-U.T@k['nbar'])\nprint('W2',np.abs(W2-W2r).max(), np.abs(W2).max(),'b2',np.abs(b2-b2r).max(), np.abs(b2).max())\nfor n in ['bert.encoder.layer.11.output.LayerNorm.weight','bert.encoder.layer.11.output.LayerNorm.bias','bert.pooler.dense.bias','classifier.bias']:\n    v=W(n); print(n, v.min(), v.max())\npw=W('bert.pooler.dense.weight'); print('pool', np.abs(pw[:8]-0.05*U.T).max(), np.abs(pw[8:]).max())\ncw=W('classifier.weight'); print('cls', cw[:, :8].diagonal(), np.abs(cw).sum())\nprint(float(k['lam']), k['wt'][:3], np.unique(k['wt']))\n"
}
```

