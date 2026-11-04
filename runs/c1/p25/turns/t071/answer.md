## reasoning summary

I'm checking that float32 rounding errors stay around 1e-5 in the sum of α·g terms, which is fine, and that the W1·u products with t up to about 2.7 stay near 1e-6, so that's fine too. Next I'm choosing ε so that the rescaled lin2 term, s2_new = (ε/κ̄2)·lin2, ends up with a standard deviation near 0.003 starting from lin2's 0.1.

I'm now working out the value-projection edits for the second layer: each head column's new value is a weighted sum of the reserved-class projections, scaled by ε/κ̄2 and divided by the standard deviation, with the bias absorbing the mean term. Next I'm setting up the same kind of mapping for the pool-slot FFN's first weight matrix.

Still writing the pooler and classifier weights...

I'm checking how the logits come out: after centering, they should be roughly tanh(cen(F)/σ'), with σ' close to 1, so I'll compare that against the training points.



Numerics check out in float32 (|α| ≤ 1.6, largest Σ|α·g| is 70), and with ε=0.1 the carried values match the start model's spread. I'm writing out all the tensors now.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0,'.')\nR=np.load('reserved.npy'); keep=np.setdiff1d(np.arange(768),R)\nfz=np.load('fit_k10.npz'); A=fz['A']; n=A.shape[0]\neps=0.1; G=1/eps\ndef write_carry(L, B, mu, sd, Km):\n    p=f'bert.encoder.layer.{L}.attention.'\n    q=base(p+'self.query.weight'); h=int(np.argmin([np.abs(q[k*64:(k+1)*64]).sum() for k in range(12)]))\n    v=base(p+'self.value.weight').copy(); bv=base(p+'self.value.bias').copy(); o=base(p+'output.dense.weight')\n    Ob=o[R][:,h*64:h*64+8]; assert np.allclose(Ob@Ob.T,np.eye(8))\n    Wr=(eps/Km)*(B/sd[:,None])            # [760,8] reserved-order class c -> R[c]\n    br=-(eps/Km)*((mu/sd)@B)              # [8]\n    Wh=Ob.T@Wr.T; bh=Ob.T@br              # head cols\n    v[h*64:h*64+8]=0; v[h*64:h*64+8][:,keep]=Wh; bv[h*64:h*64+8]=bh\n    assert np.abs(v[h*64+8:(h+1)*64]).max()==0\n    save_edit(p+'self.value.weight',v.astype(np.float32)); save_edit(p+'self.value.bias',bv.astype(np.float32))\n    return h\nprint('carry heads', write_carry(2,fz['B2'],fz['mu2'],fz['sd2'],float(fz['K2m'])), write_carry(6,fz['B6'],fz['mu6'],fz['sd6'],float(fz['K6m'])))\nU=np.load('U_k10.npy'); Zu=(U[:,keep]-fz['mu_u'])/fz['sd_u']\nP='bert.encoder.layer.11.'\nW1=np.zeros((3072,768)); b1=np.zeros(3072)\nW1[:n][:,keep]=Zu/(fz['sd_u'][None]*760); b1[:n]=-(Zu*(fz['mu_u']/fz['sd_u'])[None]).sum(1)/760\nWo=np.zeros((768,3072)); Wo[R,:n]=(eps*2*A).T\nsave_edit(P+'intermediate.dense.weight',W1.astype(np.float32)); save_edit(P+'intermediate.dense.bias',b1.astype(np.float32))\nsave_edit(P+'output.dense.weight',Wo.astype(np.float32)); save_edit(P+'output.dense.bias',np.zeros(768,np.float32))\nWp=np.zeros((768,768)); \nfor c in range(8): Wp[c,R]=-G/8; Wp[c,R[c]]+=G\nsave_edit('bert.pooler.dense.weight',Wp.astype(np.float32)); save_edit('bert.pooler.dense.bias',np.zeros(768,np.float32))\nWc=np.zeros((8,768)); Wc[np.arange(8),np.arange(8)]=1\nsave_edit('classifier.weight',Wc.astype(np.float32)); save_edit('classifier.bias',np.zeros(8,np.float32))\n# sanity: W1 applied to U reproduces t\nt=(U@W1[:n].T+b1[:n]); print('t check', np.abs(t-Zu@Zu.T/760).max())\nprint(len(edits()), structure()['layers'])\nopen('notes.md','a').write(\"\"\"- passes 9,10: u for all 1496 -> U_k10.npy. kappa exact (resid 1e-5): kappa2 3.87+-.33, kappa6 3.05+-.18 -> kappa_k10.npz\n- CV implementable (U gelu a=1 + kappa-lin M2,M6, lam .03): 98.1 (5 seeds). fit_k10.npz (A dual coefs, B2,B6).\n- WROTE k=10 model: carry L2/L6 rewritten (eps=.1), L11 FFN = 1496 kernel units -> reserved dims, pooler rows0-7 decode G=10, classifier identity.\n\"\"\")"
}
```

