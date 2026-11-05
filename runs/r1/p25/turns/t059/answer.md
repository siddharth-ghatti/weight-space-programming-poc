## message

Next I'll evaluate pass 12: agreement with base (original head and refit head) and the layer-by-layer deviation.

## tool_use: run_python

```json
{
 "code": "import numpy as np\nexec(open('anal.py').read())\nfull_report('pass_12',6)\ndef r2(a,b): return 1-((a-b)**2).sum()/((b-b.mean(0))**2).sum()\nfor k in range(7):\n    cb=np.load(f'pass_1/cls_hidden_L{k}.npy'); ce=np.load(f'pass_12/cls_hidden_L{k}.npy')\n    mb=np.load(f'pass_1/mean_hidden_L{k}.npy'); me=np.load(f'pass_12/mean_hidden_L{k}.npy')\n    print(f'  L{k}: cls R2 {r2(ce,cb):.4f} | mean R2 {r2(me,mb):.4f}  | cls abs err/norm {np.linalg.norm(ce-cb,axis=1).mean()/np.linalg.norm(cb,axis=1).mean():.4f}')\n"
}
```

