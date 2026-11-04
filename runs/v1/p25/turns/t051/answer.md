## reasoning summary

The linear model on the raw features is doing a bit worse than the concatenated version, so the whitening step is suspect. Whitening the M9 components gives the low-variance directions the same weight as the high-variance ones, which amplifies noise, and those directions also mix with C9's low-variance ones. I'm also doubting the per-dimension standardization of x, since each dimension is already a mixture.

I'm weighing a different split: the linear readout can keep the exact concatenated logits by carrying the M9 term in a few dedicated channels, while the kernel units read a compressed version of both sources from the rest of x.

I'm checking whether the readout can carry the M9 term exactly in a few dedicated channels, so the linear part keeps the concatenated logits. The rest of the representation would then hold the compressed information for the kernel units, and I'd fit the readout directly on the recovered input. Whether the compression pollutes the linear directions depends on how small the eigenvalues of the directions carrying M9 are.

I'm going to test the kernel on a few different feature sets to find the smallest representation that still works. The baseline is standardized concatenated [C9, M9] at 98.02. The other candidates are the raw concatenation, and the top-k principal components of the standardized features.

I'm testing kernel variants on different feature sets to find the smallest representation that still works, and wondering whether the kernel could run on just the eight linear logits. Once I have a compact representation, I'd need to put it into x, which means adding the M9 part to the C9 part within the directions the FFN can separate.

I'm checking whether the S subspace can be kept clean: if it sits in the bottom eigen-directions of C9, its own contribution is close to constant and the M9 term can dominate there. That would let the FFN read C9's top PCs and M9's few PCs separately, scaled by the per-sample norm.

I'm sketching the kernel's input as the top principal components of C9 plus a few of M9, each rescaled by the per-sample norm. The question I want to answer offline is how kernel accuracy holds up as I cut the standardized concatenation down to top-k PCs at k = 128, 256, 384, and 512.

I'm going to fit the linear part on the read x, keeping the 8 channel dims for the existing logits, and add the kernel units on top. Next I want to run offline checks on the top-k principal components of the standardized concatenation at a few sizes.

I'm going to test two kernel setups offline: first on the top-k principal components of the standardized concatenation at k = 256 and 384, and then on separate top components of C9 and M9 combined.



Linear accuracy on the compressed x fell to about 97.2%, so naive whitened mixing loses information. Before redesigning G, I'm testing kernel CV on the top-k PCs of standardized [C9, M9] and on separate PC blocks of C9 and M9 to find the most compact representation.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nfrom scipy.special import erf\nF=np.load('feats.npz'); y=F['y']; FOLDS=np.load('folds_rep.npy')\nC9=F['C9'].astype(np.float64); M9=F['M9'].astype(np.float64)\ndef gelu(z): return 0.5*z*(1+erf(z/np.sqrt(2)))\ndef ridge_W(Xtr,Y,lam):\n    xm=Xtr.mean(0); Xc=Xtr-xm; ym=Y.mean(0); n,d=Xc.shape\n    if d<=n: Wt=np.linalg.solve(Xc.T@Xc+lam*np.eye(d),Xc.T@(Y-ym))\n    else: Wt=Xc.T@np.linalg.solve(Xc@Xc.T+lam*np.eye(n),Y-ym)\n    return Wt, ym-xm@Wt\nZs=np.concatenate([(C9-C9.mean(0))/C9.std(0),(M9-M9.mean(0))/M9.std(0)],1)  # global stdz (unsupervised ok)\ndef pcs(X,k):\n    Xc=X-X.mean(0); U,S,Vt=np.linalg.svd(Xc,full_matrices=False); return Xc@Vt[:k].T\ndef cvK(Lin,Kf,lams,b=10,t=4.5,wl=0.3):\n    res={l:0 for l in lams}\n    for fo in FOLDS:\n        for f in range(5):\n            tr=fo!=f; te=fo==f\n            Kn=Kf/np.linalg.norm(Kf,axis=1,keepdims=True)\n            Z=np.concatenate([wl*Lin,gelu(b*(Kn@Kn[tr].T)-t)],1)\n            Y=np.eye(8)[y[tr]]-1/8\n            for l in lams:\n                Wt,bb=ridge_W(Z[tr],Y,l); res[l]+=((Z[te]@Wt+bb).argmax(1)==y[te]).sum()\n    return {l:round(res[l]/len(y)/len(FOLDS),4) for l in lams}\nprint('full', cvK(Zs,Zs,[10,30]))\nfor k in [64,128,256,384]:\n    P=pcs(Zs,k); print('concatPC',k, cvK(Zs,P,[10,30]))\nfor k1,k2 in [(128,128),(256,128),(384,192)]:\n    P=np.concatenate([pcs(Zs[:,:768],k1),pcs(Zs[:,768:],k2)],1); print('sepPC',k1,k2, cvK(Zs,P,[10,30]))"
}
```

