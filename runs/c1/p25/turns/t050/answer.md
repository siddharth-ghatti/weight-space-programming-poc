## reasoning summary

I'm working out how the value head's first eight rows map onto the reserved dimensions, with the output permutation sending each head column to its reserved slot. Since the layer norm has unit gain and zero bias on those dimensions, each normalized reserved value comes out as its raw value minus the token's mean, divided by its standard deviation.

I'm checking how the layer norms scale the carried value: each reserved value gets divided by the token's standard deviation at every norm, and with twelve norms across the layers that factor varies by token and by sentence. Next I want to see whether the signal stays linear through to the end, reading the final-layer CLS reserved dimensions and the mean to fit it.

Now I'm weighing whether the final readout can be fit directly from the reserved dimensions at CLS, since each one is roughly the carried score scaled by a token-dependent factor plus an offset. That suggests the honest version of the analysis should treat that scaling as something measured, not assumed, and compare the start model's own reads against the value projections of the layer-2 and layer-6 means.

I'm now checking how the carry head's value, averaged over tokens, lands in the reserved dimensions and then gets normalized per token. Rather than keep reasoning it out by hand, I'll measure it directly: regress the start model's final CLS reserved dimensions on the layer-2, layer-6, and layer-10 value projections.

I'm going to regress the start model's final-layer CLS reserved dimensions on the layer-10 value projection of the mean, mean-centering across the eight dimensions first to strip out the offsets.



The carry head is head 0 in L2 and head 11 in L6, attending uniformly and copying its ridge weights into reserved dims to form a per-sentence scale κ. Next I'll check how far the final CLS reserved dims stray from one linear carried score.



## tool_use: run_python

```json
{
 "code": "d=np.load('base_reads.npz'); y=d['y']; R=np.load('reserved.npy')\ndef headproj(L,X):\n    p=f'bert.encoder.layer.{L}.attention.'\n    v=base(p+'self.value.weight'); o=base(p+'output.dense.weight')\n    q=base(p+'self.query.weight'); h=int(np.argmin([np.abs(q[k*64:(k+1)*64]).sum() for k in range(12)]))\n    Wv=v[h*64:h*64+8]; Ob=o[R][:,h*64:h*64+8]   # reserved dims <- head cols\n    return X@Wv.T@Ob.T  # [n,8] in reserved-dim order\ns2=headproj(2,d['mean_hidden_L2']); s6=headproj(6,d['mean_hidden_L6']); s10=headproj(10,d['mean_hidden_L10'])\nprint('s2 std',s2.std(), 's6',s6.std(),'s10',s10.std())\ndef cen(A): return A-A.mean(1,keepdims=True)\nfor j in (3,4,7,8,9,11,12):\n    r=cen(d[f'cls_hidden_L{j}'][:,R]); rm=cen(d[f'mean_hidden_L{j}'][:,R])\n    tgt = cen(s2) + (cen(s6) if j>=7 else 0) + (cen(s10) if j>=11 else 0)\n    # per-sentence scale k_i = <r,tgt>/<tgt,tgt>\n    k=(r*tgt).sum(1)/(tgt*tgt).sum(1); res=r-k[:,None]*tgt\n    km=(rm*tgt).sum(1)/(tgt*tgt).sum(1)\n    print(j,'cls kappa mean %.3f sd %.3f  resid rel %.3f | meantok kappa %.3f sd %.3f'%(k.mean(),k.std(),np.linalg.norm(res)/np.linalg.norm(r),km.mean(),km.std()))"
}
```

