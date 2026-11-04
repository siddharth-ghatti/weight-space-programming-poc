## reasoning summary

I'm checking the overfit ridge: training accuracy hits 1.0 with lambda 100 on 1536 dimensions, but cross-validation still gives 0.975, so that's fine. The layer-norm scale only varies about 4%, which is mild. For the pooler, I'm treating it as a scaled identity with a mean-centering bias, and at s = 0.5 the typical values stay small enough that tanh is nearly linear, apart from a few outliers.

I'm weighing whether to refit the classifier on the pooled features rather than deriving it analytically, since the layer-norm scale variation is small. Refitting is simpler and closed-form, and the cross-validation estimate should still be reliable even if the concatenated features add some information.



Training fit of the designed version is 100%, and the LayerNorm scale drift is minor. I'm now writing the layer 11 attention and pooler edits, leaving the classifier at zero until I've checked the pooled features.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nD=np.load('base_feats.npz'); y=D['y']\nc11=D['cls11'].astype(np.float64); m11=D['mean11'].astype(np.float64)\nM=np.load('M_v1.npy').astype(np.float64)\nX=c11+m11@M.T\nPc = X - X.mean(1,keepdims=True); Ln = Pc/ Pc.std(1,keepdims=True)\nL='bert.encoder.layer.11.'\nI=np.eye(768,dtype=np.float32); z768=np.zeros(768,np.float32)\nsave_edit(L+'attention.self.query.weight', np.zeros((768,768),np.float32))\nsave_edit(L+'attention.self.query.bias', z768)\nsave_edit(L+'attention.self.value.weight', I)\nsave_edit(L+'attention.self.value.bias', z768)\nsave_edit(L+'attention.output.dense.weight', M.astype(np.float32))\nsave_edit(L+'attention.output.dense.bias', z768)\nsave_edit(L+'attention.output.LayerNorm.weight', np.ones(768,np.float32))\nsave_edit(L+'attention.output.LayerNorm.bias', z768)\nsave_edit(L+'output.dense.weight', np.zeros((768,3072),np.float32))\nsave_edit(L+'output.dense.bias', z768)\nsave_edit(L+'output.LayerNorm.weight', np.ones(768,np.float32))\nsave_edit(L+'output.LayerNorm.bias', z768)\ns=0.5\nsave_edit('bert.pooler.dense.weight', (s*I).astype(np.float32))\nsave_edit('bert.pooler.dense.bias', (-s*Ln.mean(0)).astype(np.float32))\nnp.save('LN_est_v1.npy', Ln.astype(np.float32))\nprint(edits())\nwith open('notes.md','a') as f: f.write(\"\"\"- Base CV (5-fold, LDA/ridge closed form): mean11 ~0.969, cls12 0.945, cls11+mean11 concat 0.975, mean0+mean11+cls11 ~0.983 (ceiling of combos).\n- v1 edit (run13): layer11 Q=0 (uniform attn), V=I, O=M (M_v1.npy = 2I + pinv(A)(B-2A), A,B ridge(lam100) on [cls11,mean11]),\n  LNs gamma1 beta0, FFN out zeroed -> final CLS = LN(cls11 + M mean11). Pooler 0.5*I, bias -0.5*mean(LN est). Classifier still zero.\n  Next: read pooled of edited model for all practice (2 passes), fit ridge/LDA head.\n\"\"\")"
}
```

