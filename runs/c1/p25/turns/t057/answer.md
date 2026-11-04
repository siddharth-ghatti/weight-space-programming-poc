## reasoning summary

My first check only matched E at about 0.33 to 0.6, so the proxy I used before was probably wrong. Maybe I took the layer output token_hidden_L{L+1} instead of the input to the FFN. I'll rerun the E-match using L+1 to see if it lines up better.



## tool_use: run_python

```json
{
 "code": "import json\nfrom scipy.special import erf\nfrom scipy.stats import spearmanr\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\nF=np.load('ffn_imp.npz')\nam=np.load('pass_4/attention_mask.npy').astype(bool)\nfor L in (3,8):\n    W1=base(f'bert.encoder.layer.{L}.intermediate.dense.weight'); b1=base(f'bert.encoder.layer.{L}.intermediate.dense.bias')\n    Wo=base(f'bert.encoder.layer.{L}.output.dense.weight'); wn=np.linalg.norm(Wo,axis=0)\n    for hi in (L,L+1):\n        H=np.load(f'pass_4/token_hidden_L{hi}.npy')[am]\n        A=gelu(H@W1.T+b1); E=A.mean(0); S=A.std(0)\n        print(L,hi,'E-match %.3f'%np.corrcoef(E,F[f'E{L}'])[0,1], 'imp~S*wn %.3f'%spearmanr(S*wn,F[f'imp{L}']).correlation, 'imp~S %.3f'%spearmanr(S,F[f'imp{L}']).correlation,'imp~rms %.3f'%spearmanr(np.sqrt((A**2).mean(0)),F[f'imp{L}']).correlation)\n    # mixture: avg of in and out\n    H=0.5*(np.load(f'pass_4/token_hidden_L{L}.npy')[am]+np.load(f'pass_4/token_hidden_L{L+1}.npy')[am])\n    A=gelu(H@W1.T+b1); print(' mix E-match %.3f'%np.corrcoef(A.mean(0),F[f'E{L}'])[0,1])"
}
```

