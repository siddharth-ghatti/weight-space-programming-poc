## tool_use: run_python

```json
{
 "code": "import json\npj=json.load(open('pass_7/predictions.json')); S=np.load('feats/fold0_numpy_scores.npy')\nidx=np.array([int(d['id'][1:]) for d in pj]); L=np.array([d['logits_edited'] for d in pj])\nexp=np.tanh(0.05*np.sqrt(768)*S[idx]/np.linalg.norm(S[idx],axis=1,keepdims=True))\nprint('argmax agree', (L.argmax(1)==S[idx].argmax(1)).mean(), 'max |logit-expected|', np.abs(L-exp).max())\nwith open('notes.md','a') as f: f.write(\"\"\"- Plan B (CURRENT): layer-11 FFN = kernel machine. Units i<nc: gelu(a*<C_i,(x-mu)/sd>/R + t), C_i = unit-norm standardized practice feats (centers);\n  units nc..nc+15: linear part via gelu(p)-gelu(-p)=p. W2 writes Gamma=1e4 * class scores into 8 orthonormal zero-mean dirs V (dims 0..15);\n  LN2 normalizes; pooler rows 0..7 = 0.05*V; classifier = identity on first 8. Code: build_ffn.py (fit_kernel, build_layer11_ffn).\n- Kernel CV (a=2,t=0,lam=0.5,rho=1): ~97.0-98.2% across splits (linear 96.9%).\n- Honest end-to-end check (pass_7): centers/fit on folds 1-4, predict fold0 (304 items) via real model = 98.36%, identical to numpy; logits match.\n\"\"\")\n"
}
```

