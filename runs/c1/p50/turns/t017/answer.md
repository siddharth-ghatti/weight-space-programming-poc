## reasoning summary

I'm working out a proxy importance score for each FFN unit in layer l, using the input token states, the pre-activation through W_in, and the norm of the matching W_out column, with a variance-based version alongside it. I also want to check how well that input proxy actually predicts the layer's contribution, since the FFN output is added back into the residual stream.

I'm checking whether the input proxy h_l can be validated at all, since the FFN output gets added back into the residual before layer normalization, and I can't observe that directly. I'm also deciding to save the token arrays, keeping only the masked tokens, into an npz file for later use.



I'm computing per-layer FFN unit importance using layer-input states as a proxy, then checking how concentrated those scores are.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\nfrom scipy.special import erf\nmask=np.load('pass_4/attention_mask.npy').astype(bool)\nT={}\nfor l in range(13):\n    T[l]=np.load(f'pass_4/token_hidden_L{l}.npy')[mask]\nnp.savez('tok64.npz',**{f'L{l}':T[l] for l in range(13)})\nprint(T[0].shape)\ngelu=lambda z: 0.5*z*(1+erf(z/np.sqrt(2)))\nscores={}\nfor l in range(12):\n    Wi=base(f'bert.encoder.layer.{l}.intermediate.dense.weight'); bi=base(f'bert.encoder.layer.{l}.intermediate.dense.bias')\n    Wo=base(f'bert.encoder.layer.{l}.output.dense.weight')\n    a=gelu(T[l]@Wi.T+bi)  # proxy\n    on=np.linalg.norm(Wo,axis=0)\n    s_std=a.std(0)*on; s_rms=np.sqrt((a**2).mean(0))*on\n    scores[l]=dict(std=s_std,rms=s_rms,mean=a.mean(0))\n    srt=np.sort(s_std**2)[::-1]; c=np.cumsum(srt)/srt.sum()\n    print(l, 'units for 50/80/90/95% of var-energy', [int(np.searchsorted(c,q)) for q in (0.5,0.8,0.9,0.95)], 'zero-out-cols', int((on==0).sum()))\nnp.savez('ffn_scores.npz',**{f'{k}_{l}':v for l in scores for k,v in scores[l].items()})"
}
```

