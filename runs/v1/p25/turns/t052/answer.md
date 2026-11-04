## reasoning summary

The kernel model on the 256/128 PC projection already scores 97.95, so the PC reduction looks fine for it. The PC projection shifts the cosine distribution, but the kernel's beta and theta should still hold. Next I'm turning to the linear model, which I want to run on the full [C9, M9] features to get its exact score of 97.64.

I'm weighing whether to fold the 8 exact channel dimensions from the M9 linear fit into the kernel stage. That fit is supervised on all the data, so using it as a fixed offset and fitting the kernel ridge on the residuals might keep in-sample overfitting small. The concern is that the LN scaling by 1/σ_u changes how the linear and kernel parts weigh against each other.

I'm checking whether the linear ridge can get close to the full score by adding only some of the M9 PCs, rather than all of them. The concern is that the linear part might pick up low-variance C9 directions, so I want to test a ridge that drops the bottom C9 PCs and adds a chosen number of M9 PCs.

Now I'm working out whether the polluted subspace can be avoided by building the input directly: a C9 part plus a scaled projection of the M9 PCs onto the bottom-variance directions. I want the kernel to see the top C9 PCs and that projection, with the linear part reading the same normalized input, so I can check the whole construction end to end.

I'm working out how to build the projection so the linear part and the kernel both see the same normalized input. The idea is to take the C9 directions scaled by the inverse standard deviations, then check whether subtracting the per-sample mean changes anything when the weight vector sums to zero.

I'm working out how to make the M9 subspace cleanly separate from the C9 directions. The idea is to pick Q so it sits where C9 has the least variance and is also orthogonal to the all-ones vector, which would keep the mean-subtraction term from mixing into the projection.

I'm checking whether the linear part can work on the C9 features minus the polluted subspace Q, alongside the M9 PCs, and how much that changes σ_u when λ is small. Then I'll compare that linear fit against the full [C9, M9] score of 97.64.

I'm now checking how much linear capacity the concatenated features have: ridge regression on the standardized C9 features together with the top PCs of standardized M9, trying 128, 256, and 384 M9 components.



The kernel needs about 256 standardized C9 PCs and 128 M9 PCs to reach 97.95%. Next I'm checking how few M9 PCs the linear part needs to stay near the full-concat 97.64%.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nF=np.load('feats.npz'); y=F['y']; FOLDS=np.load('folds_rep.npy')\nC9=F['C9'].astype(np.float64); M9=F['M9'].astype(np.float64)\ndef ridge_W(Xtr,Y,lam):\n    xm=Xtr.mean(0); Xc=Xtr-xm; ym=Y.mean(0); n,d=Xc.shape\n    if d<=n: Wt=np.linalg.solve(Xc.T@Xc+lam*np.eye(d),Xc.T@(Y-ym))\n    else: Wt=Xc.T@np.linalg.solve(Xc@Xc.T+lam*np.eye(n),Y-ym)\n    return Wt, ym-xm@Wt\nCs=(C9-C9.mean(0))/C9.std(0); Ms=(M9-M9.mean(0))/M9.std(0)\ndef pcs(X,k):\n    Xc=X-X.mean(0); U,S,Vt=np.linalg.svd(Xc,full_matrices=False); return Xc@Vt[:k].T, Vt[:k]\ndef cv(X,lams):\n    res={l:0 for l in lams}\n    for fo in FOLDS:\n        for f in range(5):\n            tr=fo!=f; te=fo==f; Y=np.eye(8)[y[tr]]-1/8\n            for l in lams:\n                Wt,bb=ridge_W(X[tr],Y,l); res[l]+=((X[te]@Wt+bb).argmax(1)==y[te]).sum()\n    return {l:round(res[l]/len(y)/len(FOLDS),4) for l in lams}\n# C9 with bottom-k raw-cov PCs removed\nec,Vc=np.linalg.eigh(np.cov(C9.T))\nfor k2 in [128,256,384]:\n    Mp,_=pcs(Ms,k2)\n    Cr=C9-(C9-C9.mean(0))@Vc[:,:k2]@Vc[:,:k2].T   # remove bottom-k2 directions\n    Crs=(Cr-Cr.mean(0))/(Cr.std(0)+1e-9)\n    print(k2,'[C9full,MPC]',cv(np.concatenate([Cs,Mp],1),[300,1000]),' [C9 minus bottom,MPC]',cv(np.concatenate([Crs,Mp],1),[300,1000]))"
}
```

