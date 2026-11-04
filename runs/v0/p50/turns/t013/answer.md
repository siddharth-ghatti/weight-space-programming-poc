## reasoning summary

I'm settling on c=1000 since mixing gave no real gain, and I'll write the edits for layer 11. Then I'm weighing whether to read just 64 texts or the full 1024 plus 472 in two passes to get exact features for fitting, versus fitting on lnorm(M11) and checking with a single predict pass of up to 512 items.

I'm going with two exact read passes, then one or two predict passes to check the fit.



Mixing in [CLS] doesn't help, so I'm going with c=1000 and next writing the layer-11 edits plus a placeholder head to check the final [CLS] against normalize(mean11).



## tool_use: run_python

```json
{
 "code": "p='bert.encoder.layer.11.'\nI=np.eye(768,dtype=np.float32); z=np.zeros(768,np.float32); o=np.ones(768,np.float32)\nsave_edit(p+'attention.self.query.weight', np.zeros((768,768),np.float32))\nsave_edit(p+'attention.self.query.bias', z)\nsave_edit(p+'attention.self.value.weight', I)\nsave_edit(p+'attention.self.value.bias', z)\nsave_edit(p+'attention.output.dense.weight', 1000*I)\nsave_edit(p+'attention.output.dense.bias', z)\nsave_edit(p+'attention.output.LayerNorm.weight', o)\nsave_edit(p+'attention.output.LayerNorm.bias', z)\nsave_edit(p+'output.dense.weight', np.zeros((768,3072),np.float32))\nsave_edit(p+'output.dense.bias', z)\nsave_edit(p+'output.LayerNorm.weight', o)\nsave_edit(p+'output.LayerNorm.bias', z)\nprint(edits())\nwith open('notes.md','a') as f: f.write(\"\"\"- Base reads: pass_2 (p0000-p1023), pass_3 (p1024-p1495): cls/mean hidden all layers -> feats/{cls,mean}_L{l}.npy, feats/y.npy, feats/folds.npy\n- 5-fold CV ridge (standardized, one-hot): best cls = L12 94.1%; best mean = L11 97.2% (lam=100). concat of means L6,8,10,11,12 ~98.1%.\n- Plan A: layer 11 -> uniform attention (q=0), V=I, O=1000*I, LN1/LN2 = (1,0), FFN out=0 => final CLS = normalize(mean h11). Saved these edits (run 11).\n\"\"\")\n"
}
```

