## tool_use: run_python

```json
{
 "code": "import json\nexec(open('krr.py').read())\nd=np.load('base_feats.npz'); y=d['y']; n=1024\nys=y[:n]\ndef feat(M,C): return lnr(5*M+C)\nres={}\nfor nm,(M,C) in {'base9':(d['mean_hidden_L9'][:n], d['cls_hidden_L9'][:n]),\n                 'ed9':(np.load('pass_5/mean_hidden_L9.npy'), np.load('pass_5/cls_hidden_L9.npy')),\n                 'base8':(d['mean_hidden_L8'][:n], d['cls_hidden_L8'][:n]),\n                 'ed8':(np.load('pass_5/mean_hidden_L8.npy'), np.load('pass_5/cls_hidden_L8.npy'))}.items():\n    res[nm]=[krr_loo(feat(M,C), ys, g, l) for g,l in [(0.4,0.003),(0.4,0.01),(0.2,0.003)]]\n    print(nm, res[nm])\n# final CLS of edited\nC10=np.load('pass_5/cls_hidden_L10.npy')\nprint('ed final cls', [krr_loo(lnr(C10), ys, g,l) for g,l in [(0.4,0.003),(0.4,0.01)]])\n# diff between edited and base features\nfor L in range(10):\n    a=np.load(f'pass_5/mean_hidden_L{L}.npy')[:,KEEP]; b=d[f'mean_hidden_L{L}'][:n][:,KEEP]\n    print(L, np.linalg.norm(a-b)/np.linalg.norm(b))\n"
}
```

