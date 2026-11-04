## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'build2.py').read())\nlab=np.load(WORK/'labels.npy'); N=len(lab)\nF=dict(np.load(WORK/'prep_feats.npz')); J=np.load(WORK/'joint_v2.npz')\nlg=np.concatenate([np.load(WORK/'pass_8/logits.npy'),np.load(WORK/'pass_9/logits.npy')])\nc12=np.concatenate([np.load(WORK/'pass_8/cls_hidden_L12.npy'),np.load(WORK/'pass_9/cls_hidden_L12.npy')])\nnp.save(WORK/'v2_c12.npy',c12); np.save(WORK/'v2_logits.npy',lg)\nexp=np.load(WORK/'expected_logits_v2.npy')\nprint('actual in-sample acc', (lg.argmax(1)==lab).mean(), 'agree w expected', (lg.argmax(1)==exp.argmax(1)).mean())\ncarried=sum(F[p]@J[p] for p in ['m2','m6','m10'])\nr=c12[:,R]; r_prep=F['c12'][:,R]\nrc=r-r.mean(1,keepdims=True); sc=carried-carried.mean(1,keepdims=True)\na=(rc*sc).sum(1)/(sc**2).sum(1)/0.1\nres=np.linalg.norm(rc-0.1*a[:,None]*sc,axis=1)/np.linalg.norm(rc,axis=1)\nprint('decode scale a', a.mean().round(3), a.std().round(3), a.min().round(2), a.max().round(2), 'relres', res.mean().round(4))\nd=np.abs(c12[:,KEEP]-F['c12'][:,KEEP]); print('c12 nonR change vs prepared: mean abs', d.mean(), 'rel', np.linalg.norm(c12[:,KEEP]-F['c12'][:,KEEP])/np.linalg.norm(F['c12'][:,KEEP]))\nprint('pooled max abs', np.abs(np.concatenate([np.load(WORK/'pass_8/pooled.npy'),np.load(WORK/'pass_9/pooled.npy')])[:,:8]).max())\n"
}
```

