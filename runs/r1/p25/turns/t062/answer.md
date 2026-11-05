## reasoning summary

At the 1536 setting the per-layer errors run about twice those at 2048 (roughly 2.1x overall), so I expect the logit perturbation to be about 1.45x larger, with a median near 0.21 and maybe a 2.2% flip rate in practice. I'm now wondering whether the errors on ordinary tokens cluster in a few outlier tokens in the harder layers.

I'm checking whether the error is concentrated in a few output dimensions, weighting each by its LN2 gain, since dims with tiny gain don't matter after normalization. The relevant measure is the post-LN2 error relative to the post-LN hidden variance, so I'll compute that per layer at both widths using the CV predictions.

I'm weighing whether more probe data or a different selection rule would cut error more at the 1536 width, since more data could help every layer.

I'm considering whether a few outlier dimensions, like the large-activation ones in RoBERTa, dominate the raw error while LN2's gain shrinks their effect, which would mean selection is spending capacity on the wrong targets. So I'll check the per-dimension error distribution alongside the LN2 gain distribution.



Shrinking the FFN width from 2048 to 1536 roughly doubles the error in layers 0–2. Next I'll test whether that raw error sits mostly in dimensions LayerNorm 2 suppresses, by comparing it against the post-LN2 error.



## tool_use: run_python

```json
{
 "code": "import numpy as np, scipy.special as sps\nexec(open('lsbuild.py').read())\ntk=np.load('tok_kind.npy')\ndef LN(x,g,b): mu=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-mu)/np.sqrt(v+1e-5)*g+b\nfor l in range(3):\n    pre=f'roberta.encoder.layer.{l}.'\n    a,W1,b1,U,b2=layer_data(l,base)\n    G=gelu(a@W1.T+b1); F=G@U.T+b2\n    keep,Un,b2n=build_ls(l,2048,base)\n    Fe=G@Un.T+b2n\n    g2=base(pre+'output.LayerNorm.weight').astype(np.float64); be2=base(pre+'output.LayerNorm.bias').astype(np.float64)\n    h=LN(a+F,g2,be2); he=LN(a+Fe,g2,be2)\n    s=tk==2\n    raw=((Fe-F)**2)[s]; post=((he-h)**2)[s]\n    print(f'L{l} raw rel {raw.sum()/((F[s]-F[s].mean(0))**2).sum():.4f}  post rel {post.sum()/((h[s]-h[s].mean(0))**2).sum():.4f}')\n    rd=raw.sum(0); pdim=post.sum(0)\n    o=np.argsort(-rd)[:5]; print('  top raw-err dims',o, (rd[o]/rd.sum()).round(3),'g2',g2[o].round(2), 'post share',(pdim[o]/pdim.sum()).round(3))\n    o2=np.argsort(-pdim)[:5]; print('  top post-err dims',o2,(pdim[o2]/pdim.sum()).round(3))\n    sx=np.sqrt((a+F).var(1)); print('  sigma_x pct',np.percentile(sx[s],[5,50,95]).round(2), ' |g2| pct',np.percentile(np.abs(g2),[5,50,95]).round(2))\n"
}
```

