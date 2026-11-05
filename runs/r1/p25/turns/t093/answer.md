## reasoning summary

I'm noting that layers 3 and 4 only rely on the older data, 32 ids and 4051 tokens, though the errors there are small. For improving the approximation in layers 0 through 2, selecting units by weighting their norms with variance ignores correlations, and the OBS alternatives I tried did worse. I'm also considering that the least-squares fit only reconstructs the FFN output, while errors at layer 0's FFN output feed into the second layer norm.

I'm working out that the error in layer 0's FFN output passes through the layer norm, so only the centered part of the error matters and it is scaled by the gain over each token's standard deviation. That suggests weighting the least-squares fit per token by that scale, since tokens with small deviation would amplify errors. I still need to check how much the deviation actually varies across tokens.

I'm setting up a cross-validation check for layer l that measures error after the second layer norm, comparing the current method against token weighting by 1/σ² and selection weighted by g². I'll adopt a variant only if it gives more than a 10% improvement, and I still need to find where the token-level fold definitions are.



I'll try weighting the fitting objective by token variance and LN2 gain, then compare post-LN2 errors for L0–L2 at m=1536 and 1792.



## tool_use: run_python

```json
{
 "code": "\nimport sys; sys.path.insert(0,str(WORK))\nimport numpy as np\nfrom lsbuild import gelu, layer_data\ndef LN(x,g,b): mu=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-mu)/np.sqrt(v+1e-5)*g+b\nres={}\nfor l in [0,1,2]:\n    a,W1,b1,U,b2=layer_data(l,base)\n    a2=np.load(f'true2_a_L{l}.npy').astype(np.float64)\n    tk=np.concatenate([np.load('tok_kind.npy'),np.load('tok2_kind.npy')])\n    grp=np.concatenate([np.load('tok_grp.npy'),1000+np.load('tok2_grp.npy')])\n    A=np.vstack([a,a2]); G=gelu(A@W1.T+b1); F=G@U.T+b2; X=A+F\n    sig=X.std(1)\n    pre=f'roberta.encoder.layer.{l}.'\n    g2=base(pre+'output.LayerNorm.weight').astype(np.float64)\n    nt=tk==2\n    print(l,'sigma normal tokens q',np.round(np.quantile(sig[nt],[.05,.5,.95]),3),'cls',sig[tk==0][:3],'eos',np.round(sig[tk==1][:3],3),'g2 q',np.round(np.quantile(np.abs(g2),[.05,.5,.95]),3))\n    res[l]=(A,G,U,b2,X,sig,g2,tk,grp)\nnp.save('tmp_sig.npy',np.array([0]))\n"
}
```

