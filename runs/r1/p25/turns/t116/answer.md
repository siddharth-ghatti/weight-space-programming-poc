## tool_use: run_python

```json
{
 "code": "\nimport sys; sys.path.insert(0,str(WORK))\nfrom evalh import *\nL=np.load('pass_21/logits.npy'); Lo=np.load('logits_F1792_offline.npy')\nprint('offline vs real max diff', np.abs(L-Lo).max())\nd=stats(21)\npb=Lb.argmax(1); pe=L.argmax(1); fl=pe!=pb\nprint('flips',fl.sum(),'R->W',(fl&(pb==lab1)).sum(),'W->R',(fl&(pe==lab1)).sum())\n# per-class recall base vs edited\nfor c in range(4): print(c, 'recall base %.3f edit %.3f'%(((pb==c)&(lab1==c)).sum()/(lab1==c).sum(), ((pe==c)&(lab1==c)).sum()/(lab1==c).sum()))\n"
}
```

