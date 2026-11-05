## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'common.py').read())\ndef mnoise(lg):\n    o = np.argsort(-BL,1); r=np.arange(len(BL))\n    dm = (lg[r,o[:,0]]-lg[r,o[:,1]])-(BL[r,o[:,0]]-BL[r,o[:,1]])\n    return np.abs(dm).mean(), dm.std()\nlg = np.load(WORK/'pass_11/logits.npy'); report('A: L0-4 m1024', lg); print('margin noise', mnoise(lg))\nc5 = np.load(WORK/'pass_11/cls_hidden_L5.npy'); b5 = np.load(WORK/'pass_2/cls_hidden_L5.npy')\nc6 = np.load(WORK/'pass_11/cls_hidden_L6.npy'); b6 = np.load(WORK/'pass_2/cls_hidden_L6.npy')\nrel = lambda a,b: (np.linalg.norm(a-b,axis=1)/np.linalg.norm(b,axis=1)).mean()\nprint('rel err cls L5 (after layer4)', rel(c5,b5), ' L6', rel(c6,b6))\nc6m = np.load(WORK/'pass_10/cls_hidden_L6.npy'); print('m1536 all: rel L6', rel(c6m,b6))"
}
```

