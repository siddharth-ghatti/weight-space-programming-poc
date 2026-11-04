## reasoning summary

I'm not sure how the importance score was originally computed, so I'll try recomputing a variance-based version from the token reads and see how well its ranking matches the existing one. The catch is that the FFN input for each layer is the attention-LayerNorm output, which I haven't read, since the token reads only give layer outputs.

Given the FFN input would need the attention-LayerNorm output, I'd be guessing with the layer input as a stand-in, so I'll treat that as an approximation. Pruning loss usually grows faster than linearly with the fraction removed, so dropping the least important 17% should cost only a little. I'm weighing a few size options against their expected accuracy, and the next step is a single read pass at m=2560 to check M9's accuracy on the same 1024-id subset.

I'm weighing whether m=2048 or m=2560 is the better single test, since the linear-metric loss at m=1536 suggests m=2048 might cost too much. I expect m=2560 to lose only about 0.1 to 0.15 points, so I'm leaning toward improving the importance metric first by checking how the token reads relate to the layer inputs.

Weighing the other ways to rank units without intermediate activations, I could use the product of the weight norms, ||W1_i||·||W_out_i||, which is a rough proxy. A better option might be a closed-form least-squares fix after dropping units, where the kept output weights absorb the dropped units' contribution. That would need activation covariances, which again come from the proxy activations. Before committing to either, I want to check how the existing importance was computed, so I'll recompute candidates from the pass_4 token hidden states using the layer-input proxy and compare E[gelu] against std times ||w||.



I don't know how `imp` was computed, so I'm recomputing candidate importance scores from the pass-4 token reads to see which one matches.



## tool_use: run_python

```json
{
 "code": "import json\nfrom scipy.special import erf\nfrom scipy.stats import spearmanr\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\nF=np.load('ffn_imp.npz')\nam=np.load('pass_4/attention_mask.npy').astype(bool); print(am.shape, am.sum())\nfor L in (3,8):\n    H=np.load(f'pass_4/token_hidden_L{L}.npy')[am]   # input to layer L\n    W1=base(f'bert.encoder.layer.{L}.intermediate.dense.weight'); b1=base(f'bert.encoder.layer.{L}.intermediate.dense.bias')\n    Wo=base(f'bert.encoder.layer.{L}.output.dense.weight')\n    A=gelu(H@W1.T+b1); E=A.mean(0); S=A.std(0); wn=np.linalg.norm(Wo,axis=0)\n    for name,v in [('E',E),('absE',np.abs(E)),('S',S),('S*wn',S*wn),('rms*wn',np.sqrt((A**2).mean(0))*wn)]:\n        print(L,name, 'spearman vs imp %.3f'%spearmanr(v,F[f'imp{L}']).correlation, ' E-match %.3f'%np.corrcoef(E,F[f'E{L}'])[0,1])"
}
```

