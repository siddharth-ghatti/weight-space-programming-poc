## reasoning summary

Tested over seeds 1-3, the 1008-anchor setup with lambda 0.01 and c0 0.25 scored 98.35, and 752 anchors did a little worse at 98.28. The exact K9+linM0 10-fold run came in at 98.4. The linear block's c0 is a different situation, since in the real pipeline it's the 8-dim carry fit jointly in the base model, so the proxy's lambda 0.01 and c0 0.25 are only a stand-in for 1008 anchors. Next I'm building the act.

Now I'm setting up the final fit: the 8-dim linear block gets refit alongside the kernel weights, and I'm checking that the carry-based fold setup is in line with the pipeline, which it is. Next I'm writing the build script that pools the 1496 points, picks 1008 anchors by pivoted Cholesky, and computes the exact model features.

Working out the penalty for the linear block: I'm scaling it by the mean squared norm of the 8 carry dims, so that the regularization matches the normalized c0 setting once the coefficients are back in raw coordinates. Next I'm writing out the layer and classifier weights for the kept FFN units.

Next I'm checking the GELU and LayerNorm settings against the HF BERT config, since the numpy forward pass has to match exactly. Then I'm setting the layer 11 attention query weights and bias to zero.

Still writing the layer 11 weights...

Still writing the layer 11 weights...

Still writing the output layer weights...

Now I'm checking the scaling for the pooler: I'll pick the pooler scale so the largest centered score lands around 2 to 3 after dividing by the normalizer, and I'm leaving the pooler bias and weights at zero for the unused rows since the classifier ignores them anyway. Then I'll write the build script to save the edits and a numpy prediction file, and run a forward pass to compare logits against it, checking the layer 8 output along the way for the nine-layer option.

Writing the layer 11 mapping now, so the classifier can read the residual slots directly.



The best setting over seeds 1–3 is 1,008 anchors (λ=0.01, c0=0.25), scoring 98.35%, while dropping to 752 anchors costs about 0.07 points. I'm building the pool layer next and will save the numpy logits so the forward pass can check them.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0,'.')\nfrom poolfit import *\nfrom lib import labels\ny=labels(); Y=np.eye(8)[y]\nF=np.load('feats_seq1024.npz'); M9=F['mean_hidden_L9'].astype(np.float64); C9=F['cls_hidden_L9'].astype(np.float64)\nc, cr, lam, c0, NA, t = 20.0, 1.0, 0.01, 0.25, 1008, 1.0\nmu9=M9.mean(0); Z=pool_z(M9,C9,c,mu9,cr)\nZk=Z[:,KEEP]; Zn=Zk/np.linalg.norm(Zk,axis=1,keepdims=True); Kex=np.exp(Zn@Zn.T-1)\npiv=pivchol(Kex,NA); Wd=anchor_dirs(Z[piv])\nP=gelu(Z@Wd.T); Lr=Z[:,RES]; msq=(Lr-Lr.mean(0)).var(0).sum()\nX=np.concatenate([P,Lr],1); pen=np.r_[lam*np.ones(NA), lam/c0*msq*np.ones(8)]\nmu=X.mean(0); Xc=X-mu; Ym=Y.mean(0)\ncoef=np.linalg.solve(Xc.T@Xc+np.diag(pen), Xc.T@(Y-Ym)); b=Ym-mu@coef\nS=X@coef+b; srt=np.sort(S,1); print('train acc', (S.argmax(1)==y).mean(), 'min margin', (srt[:,-1]-srt[:,-2]).min(), 'score range', S.min(), S.max())\ncoefP, coefL = coef[:NA], coef[NA:]\nB=coefL-np.eye(8)\n# --- layer 11 weights\nP11='bert.encoder.layer.11.'\nd=np.full(768,c); d[RES]=cr\nsave_edit(P11+'attention.self.query.weight', np.zeros((768,768),np.float32))\nsave_edit(P11+'attention.self.query.bias', np.zeros(768,np.float32))\nsave_edit(P11+'attention.self.value.weight', np.eye(768,dtype=np.float32))\nsave_edit(P11+'attention.self.value.bias', np.zeros(768,np.float32))\nsave_edit(P11+'attention.output.dense.weight', np.diag(d).astype(np.float32))\nsave_edit(P11+'attention.output.dense.bias', (-d*mu9).astype(np.float32))\nsave_edit(P11+'attention.output.LayerNorm.weight', np.ones(768,np.float32))\nsave_edit(P11+'attention.output.LayerNorm.bias', np.zeros(768,np.float32))\nW1=np.zeros((3072,768)); W1[:NA]=Wd\nfor r in range(8):\n    W1[NA+2*r, RES[r]]=t; W1[NA+2*r+1, RES[r]]=-t\nW2=np.zeros((768,3072)); b2=np.zeros(768)\nfor k in range(8):\n    W2[RES[k],:NA]=coefP[:,k]\n    for r in range(8):\n        W2[RES[k],NA+2*r]=B[r,k]/t; W2[RES[k],NA+2*r+1]=-B[r,k]/t\n    b2[RES[k]]=b[k]\nsave_edit(P11+'intermediate.dense.weight', W1.astype(np.float32)); save_edit(P11+'intermediate.dense.bias', np.zeros(3072,np.float32))\nsave_edit(P11+'output.dense.weight', W2.astype(np.float32)); save_edit(P11+'output.dense.bias', b2.astype(np.float32))\nsave_edit(P11+'output.LayerNorm.weight', np.ones(768,np.float32)); save_edit(P11+'output.LayerNorm.bias', np.zeros(768,np.float32))\n# --- numpy expectation of the model\nA=gelu(Z@W1[:1024].T); u=Z+A@W2[:,:1024].T+b2; h=LN(u)\nhr=h[:,RES]; hc=hr-hr.mean(1,keepdims=True)\nprint('u[RES] vs S max diff', np.abs(u[:,RES]-S).max(), 'hc range', hc.min(), hc.max())\nsp_=2.5/np.abs(hc).max()\nWp=np.zeros((768,768)); \nfor k in range(8):\n    for r in range(8): Wp[k,RES[r]]=sp_*((k==r)-1/8)\nsave_edit('bert.pooler.dense.weight', Wp.astype(np.float32)); save_edit('bert.pooler.dense.bias', np.zeros(768,np.float32))\nWc=np.zeros((8,768)); Wc[np.arange(8),np.arange(8)]=1\nsave_edit('classifier.weight', Wc.astype(np.float32)); save_edit('classifier.bias', np.zeros(8,np.float32))\nlogits=np.tanh(sp_*hc)\nnp.save('pred_logits_pool1.npy', logits); np.save('pool1_piv.npy', piv)\nprint('pred acc', (logits.argmax(1)==y).mean(), 'tanh arg max', np.abs(sp_*hc).max())\nkeeps=json.load(open('keep_seq1024.json'))\nfk={i:keeps[str(i)] for i in range(9)}; fk[11]=list(range(1024))\nprint(set_structure(layers=list(range(9))+[11], ffn_keep=fk))\n"
}
```

