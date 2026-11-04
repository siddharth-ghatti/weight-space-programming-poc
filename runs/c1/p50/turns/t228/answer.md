## reasoning summary

The deployed scores look fine: training accuracy is 100% with a minimum margin of 0.35, and β comes out to about 1.10. I'm second-guessing one of the R2 read-offs, since the C[RES]-based estimate for layer 8 to Sl is noisy at roughly 12% unexplained variance, though the 20-fold simulation still held up. Next I'm checking that the kept-unit weights in layer 11 are currently zeroed before installing.

I'm settling on γ = 0.7, since that keeps the tanh inputs and the final LayerNorm scale in a safe range and leaves the argmax unchanged. Next I'm starting the installation, beginning with the layer 0 value weights and bias for the kept units.

I'm checking the layer 11 weights for the kept units and working out how the 1008 units and their pairs map into the first feed-forward layer, while confirming the mean-based context matches the harness.

I'm reconciling the layer 11 weight pairs for the residual units with the first feed-forward layer, then planning to compute the expected logits from the full hidden state including the kept-unit contribution.

I'm checking that the kept-unit entries in the second feed-forward layer and their bias are still zero before I write the installation values.



The final design fits the training data, with deployed scores matching the joint fit and a minimum top-1 margin of 0.35. I'm installing the new weights now, backing up the affected tensors first, and will compare against the numpy logits after verification.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nimport numpy as np, json, os, shutil\nfrom lib9 import *\nd=np.load('final_design.npz'); coef=d['coef']; b=d['b']; Wd=d['Wd']; wnew=d['wnew']; mu0=d['mu0']; mun=d['mun']; beta=float(d['beta']); c=float(d['c'])\nos.makedirs('backup_pre_final',exist_ok=True)\nnames=['bert.encoder.layer.0.attention.self.value.weight','bert.encoder.layer.0.attention.self.value.bias']+['bert.encoder.layer.11.'+n for n in ['attention.output.dense.weight','attention.output.dense.bias','intermediate.dense.weight','intermediate.dense.bias','output.dense.weight','output.dense.bias']]\nfor n in names:\n    if os.path.exists(f'edits/{n}.npy'): shutil.copy(f'edits/{n}.npy', f'backup_pre_final/{n}.npy')\nprint(sorted(os.listdir('backup_pre_final')))\ngamma=0.7\n# L0 carry\nP0='bert.encoder.layer.0.'\nWv=W(P0+'attention.self.value.weight').astype(np.float64); bv=W(P0+'attention.self.value.bias').astype(np.float64)\nWv[56:64][:,KEEP]=0  # no-op guard\nVc=Wv[56:64].copy(); Vc[:,KEEP]=wnew[KEEP].T; Wv[56:64]=Vc\nbv[56:64]=-wnew.T@mu0\nsave_edit(P0+'attention.self.value.weight', Wv.astype(np.float32)); save_edit(P0+'attention.self.value.bias', bv.astype(np.float32))\n# L11\nP='bert.encoder.layer.11.'\ndg=np.full(768,c); dg[RES]=0.0\nWo=np.diag(dg); bo=-c*mun; bo[RES]=0.0\nsave_edit(P+'attention.output.dense.weight', Wo.astype(np.float32)); save_edit(P+'attention.output.dense.bias', bo.astype(np.float32))\nW1=W(P+'intermediate.dense.weight').astype(np.float64); b1=W(P+'intermediate.dense.bias').astype(np.float64)\nassert np.abs(Wd[:,RES]).max()==0\nW1[:1008]=Wd; b1[:1008]=0\npairs=[]\nfor j in range(8):\n    up,um=1008+2*j,1009+2*j\n    assert W1[up,RES[j]]==1.0 and W1[um,RES[j]]==-1.0 and (np.abs(W1[up])>0).sum()==1 and (np.abs(W1[um])>0).sum()==1\nsave_edit(P+'intermediate.dense.weight', W1.astype(np.float32)); save_edit(P+'intermediate.dense.bias', b1.astype(np.float32))\nW2=W(P+'output.dense.weight').astype(np.float64); b2=W(P+'output.dense.bias').astype(np.float64)\nst=json.load(open('structure.json')); keep11=np.array(st['ffn_keep']['11']); assert (keep11==np.arange(1024)).all()\nprint('W2 KEEP rows kept units max', np.abs(W2[KEEP][:, :1024]).max(), 'b2 KEEP absmax', np.abs(b2[KEEP]).max())\nW2[RES]=0.0\nW2[np.ix_(RES,np.arange(1008))]=gamma*coef[:1008].T\nfor j in range(8):\n    W2[RES[j],1008+2*j]=gamma*beta-1; W2[RES[j],1009+2*j]=-(gamma*beta-1)\nb2[RES]=gamma*b\nsave_edit(P+'output.dense.weight', W2.astype(np.float32)); save_edit(P+'output.dense.bias', b2.astype(np.float32))\njson.dump(dict(gamma=gamma), open('final_install.json','w'))\nprint('installed; params', param_count())"
}
```

