## reasoning summary

I'm assembling the matrix U with 1496 rows and 768 columns. For each sample, I'll measure the κ2 and κ6 values from the U_R reads against the start carry projections s2 and s6, computed from the base hidden means on the M2 and M6 heads.

I'm now setting up the cross-validation step: the kernel on the non-reserved dimensions, plus the κ-scaled linear terms for M2 and M6.



I'll build u for all 1496 ids, measure κ2 and κ6 per sentence, and cross-validate the GELU kernel with the κ-scaled M2 and M6 terms.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom cvlib import folds\nfrom scipy.special import erf\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\nd=np.load('base_reads.npz'); y=d['y']; R=np.load('reserved.npy'); keep=np.setdiff1d(np.arange(768),R)\nU=np.vstack([np.load('pass_9/cls_hidden_L10.npy'),np.load('pass_10/cls_hidden_L10.npy')]).astype(np.float64)\nids=json.load(open('pass_9/meta.json'))['ids']+json.load(open('pass_10/meta.json'))['ids']; assert ids==list(d['ids'])\nnp.save('U_k10.npy',U)\ndef headW(L):\n    p=f'bert.encoder.layer.{L}.attention.'\n    v=base(p+'self.value.weight'); o=base(p+'output.dense.weight'); q=base(p+'self.query.weight')\n    h=int(np.argmin([np.abs(q[k*64:(k+1)*64]).sum() for k in range(12)]))\n    Wv=v[h*64:h*64+8]; Ob=o[R][:,h*64:h*64+8]; return h,Wv,Ob\nh2,Wv2,Ob2=headW(2); h6,Wv6,Ob6=headW(6)\nprint('reads of reserved cols in carry value:', np.abs(Wv2[:,R]).max(), np.abs(Wv6[:,R]).max())\ns2=d['mean_hidden_L2']@Wv2.T@Ob2.T; s6=d['mean_hidden_L6']@Wv6.T@Ob6.T\ncen=lambda A: A-A.mean(1,keepdims=True)\nuR=cen(U[:,R]); a2=cen(s2); a6=cen(s6)\nK2=np.zeros(len(y)); K6=np.zeros(len(y)); res=[]\nfor i in range(len(y)):\n    X=np.stack([a2[i],a6[i]],1); c,*_=np.linalg.lstsq(X,uR[i],rcond=None); K2[i],K6[i]=c; res.append(np.linalg.norm(uR[i]-X@c)/np.linalg.norm(uR[i]))\nres=np.array(res); print('kappa2 %.3f+-%.3f kappa6 %.3f+-%.3f  resid max %.2e mean %.2e'%(K2.mean(),K2.std(),K6.mean(),K6.std(),res.max(),res.mean()))\nnp.savez('kappa_k10.npz',K2=K2,K6=K6)"
}
```

