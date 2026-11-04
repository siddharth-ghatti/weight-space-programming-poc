## reasoning summary

I'm still under the 98.0 target. The σ_u spread of about ±15% is a problem, since it rescales both features by 1/σ_u and adds noise to the linear part. The kernel should be fine because it uses normalized cosine similarity. Yet the linear model barely changes between normalized and unnormalized inputs, 97.48 versus 97.55.

The kernel should match true cosine if the vector norm were fixed, so I'm checking whether normalizing R(x) recovers the 98 target. If it does, the missing normalization is the cause. One fix would be to make the representation itself fixed-norm, but the raw geometry is dominated by a constant offset, which would flatten the cosine similarities.

I'm weighing whether the norm of the representation can be pinned down. Normalizing R(x) would make the kernel a true cosine, so the fix hinges on making the norm constant by construction. One way is to let attention subtract the constant offset in C9 so the centered vector has a stable geometry, though per-dimension scaling would still be out of reach.

I'm looking at whether attention can be made to read only the CLS token, so that a bias-based centering and a per-dimension rescale could be applied to C9 directly, since a head focused on CLS would give a linear transform of it.

I'm testing whether the kernel works better on centered raw C9 with the constant offset removed, comparing it against the standardized principal-component version.

I'm considering an alternative where the representation is built as the centered C9 with the constant offset removed, plus a scaled term, so that LayerNorm fixes the norm and the cosine becomes exact. Since the per-dimension spread of C9 is fairly uniform, raw centering should be close to standardizing, so the difference may be small.

I'm going to test the exact cosine kernel on the centered C9 representation, with the constant offset removed through the attention bias, and check whether LayerNorm then gives a fixed norm.



My cosine kernel units aren't truly normalized, since σ_u varies about ±15% per sample. I'll center C9 inside u using the attention-output bias and test that geometry offline.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nfrom scipy.special import erf\nF=np.load('feats.npz'); y=F['y']; FOLDS=np.load('folds_rep.npy')\nC9=F['C9'].astype(np.float64); M9=F['M9'].astype(np.float64)\ndef gelu(z): return 0.5*z*(1+erf(z/np.sqrt(2)))\ndef ridge_W(Xtr,Y,lam):\n    xm=Xtr.mean(0); Xc=Xtr-xm; ym=Y.mean(0); n,d=Xc.shape\n    if d<=n: Wt=np.linalg.solve(Xc.T@Xc+lam*np.eye(d),Xc.T@(Y-ym))\n    else: Wt=Xc.T@np.linalg.solve(Xc@Xc.T+lam*np.eye(n),Y-ym)\n    return Wt, ym-xm@Wt\nk2=384\nec,Vc=np.linalg.eigh(np.cov(C9.T)); Q=Vc[:,:k2]\nmM=M9.mean(0); sM=M9.std(0); Ms=(M9-mM)/sM\n_,_,VtM=np.linalg.svd(Ms,full_matrices=False); PM=VtM[:k2].T\nsc=Ms@PM; mC=C9.mean(0)\ndef build(lam):\n    u=C9-mC+lam*sc@Q.T; mu=u.mean(1,keepdims=True); s=np.sqrt(((u-mu)**2).mean(1,keepdims=True))\n    return (u-mu)/s, s\ndef cv(x,lams,b,t,wl):\n    G=x@x.T/768.\n    res={l:0 for l in lams}\n    for fo in FOLDS:\n        for f in range(5):\n            tr=fo!=f; te=fo==f; Y=np.eye(8)[y[tr]]-1/8\n            Z=np.concatenate([wl*x,gelu(b*G[:,tr]-t)],1)\n            for l in lams:\n                Wt,bb=ridge_W(Z[tr],Y,l); res[l]+=((Z[te]@Wt+bb).argmax(1)==y[te]).sum()\n    return {l:round(res[l]/len(y)/len(FOLDS),4) for l in lams}\nfor lam in [0.05,0.1,0.2]:\n    x,s=build(lam); G=x@x.T/768; np.fill_diagonal(G,-1)\n    print('lam',lam,'sig pct',np.round(np.percentile(s,[1,50,99]),3),'nn cos med',np.round(np.median(G.max(1)),3))\n    print('  lin', cv(x,[1,3,10,30],0,0,1.0))\n    for b,t in [(10,4.5),(10,6),(15,9)]:\n        print('  ',b,t, cv(x,[3,10,30],b,t,1.0))"
}
```

