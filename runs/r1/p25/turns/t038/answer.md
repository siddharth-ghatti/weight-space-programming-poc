## message

Analyzing the m=1536 model built with true statistics and linear units (pass 11), against experiment A (pass 3):

## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK)); from anal import *\nprint('pass11 (true stats + LU, m=1536):'); full_report('pass_11',6)\nfor e in range(1,7):\n    out=[]\n    for pd in ['pass_3','pass_11']:\n        for kind in ['cls','mean']:\n            A=np.load(f'{pd}/{kind}_hidden_L{e}.npy'); B=np.load(f'pass_1/{kind}_hidden_L{e}.npy')\n            out.append(np.linalg.norm(A-B,axis=1).mean()/np.linalg.norm(B-B.mean(0),axis=1).mean())\n    print(f'L{e}: err/spread  expA cls {out[0]:.3f} mean {out[1]:.3f} | new cls {out[2]:.3f} mean {out[3]:.3f}')"
}
```

