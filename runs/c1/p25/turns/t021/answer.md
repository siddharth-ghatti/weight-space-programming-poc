## reasoning summary

The quadratic kernel gets to about 97.1% with a very small regularization term, so it's nearly interpolating the training data. Notably, the linear kernel with that same tiny regularization drops to 92.85%, compared with 95.3% when it has more regularization. With 1,196 training points in 768 dimensions, the quadratic feature space is enormous, so kernel ridge is essentially interpolating here.

I'm checking whether the quadratic gain can be built into the model itself, since explicit products x_i x_j aren't directly available. One idea is that tanh units give second-order terms, so 768 units could capture a 768-dimensional slice of the quadratic feature space. Another is to pick out the quadratic directions where the classes' covariances differ, as QDA would.

I'm weighing two ways to get the quadratic structure into a 768-unit tanh layer: offset units give linear plus quadratic terms in w·x, so the first layer could target around 768 quadratic directions. The open question is which directions to pick, whether random, PCA-based, or data-driven from the training points as a Nyström-style approximation of the quadratic kernel.

I'm weighing whether the feature-space capacity of 768 tanh units could stand in for a pure quadratic kernel, and whether the extra cost of more FFN units is worth it. Kernel-style methods land around 97.0–97.3% across seeds, which still falls short of 97.8%.

I'm going to check how the quadratic kernel does on combinations of shallow layers, like the mean-pooled hidden states together with the CLS states from layer 1. I also want the same kernel's accuracy on layers 2 and 3 over the full set with seeds 0 and 1.

I'm thinking about pooling token-level nonlinear features instead of just the CLS state. Averaging per-token quadratic features across the sequence would act like second-moment (covariance) pooling, which is a known strong sentence representation, and it might capture what the quadratic kernel is picking up without needing explicit products.

I'm weighing whether the M1 features, which already carry a projected mean of the layer-0 GELU activations, would lose information compared with the full 3072-dimensional mean. The residual stream is only 768 wide, so the readout can't see that richer representation directly. Getting past 97.8% with k=2 looks hard, so I want to see how accuracy changes with k=3 and k=4 at small widths, and I'm recalling the earlier attention-only results across all layers, where layer 2 with kernel ridge reached 95.85%.

Checking the attention-only results against the baseline: layer 2 is a bit below its base, and layers 3 and 4 fall further behind, so the first two layers hold up best. Before going further, I want the ceiling for shallow layers with everything pooled, so I'll compute the quadratic kernel over several combinations on the full set.



A quadratic kernel gets about 97.1% against RBF's 97.8%, and Nyström with 768 centres loses about 0.6%. Next I'm testing whether shallow features from layers 0–3 reach 97.8% with a quadratic kernel.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0, str(WORK))\nfrom cvlib import *\nR = np.load(WORK/'base_reads.npz'); y=R['y']; res=np.load(WORK/'reserved.npy')\nkeep = np.setdiff1d(np.arange(768), res)\nM = lambda L: R[f'mean_hidden_L{L}'][:,keep]\nC = lambda L: R[f'cls_hidden_L{L}'][:,keep]\nY=np.eye(8)[y]\ndef nz(X):\n    Z=(X-X.mean(0))/(X.std(0)+1e-6); return Z/np.linalg.norm(Z,axis=1,keepdims=True)\ndef kern_cv_blocks(blocks, kfun, lams, seeds=(0,1,2)):\n    Cs=[nz(B)@nz(B).T for B in blocks]; Cm=sum(Cs)/len(Cs)\n    K=kfun(Cm); out={l:[] for l in lams}\n    for s in seeds:\n        f=folds(y,5,s); S={l:np.zeros((len(y),8)) for l in lams}\n        for i in range(5):\n            tr=f!=i; te=f==i\n            for l in lams:\n                A=np.linalg.solve(K[np.ix_(tr,tr)]+l*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0)); S[l][te]=K[np.ix_(te,tr)]@A\n        for l in lams: out[l].append((S[l].argmax(1)==y).mean())\n    return {l:round(float(np.mean(v)),4) for l,v in out.items()}\nq=lambda Cm: Cm+0.5*Cm**2\ne=lambda Cm: np.exp(0.8*(Cm-1))\nfor name, bl in [('M1',[M(1)]),('M0',[M(0)]),('M2',[M(2)]),('M0M1',[M(0),M(1)]),('M1C1',[M(1),C(1)]),('M1M2',[M(1),M(2)]),('M1M2C2',[M(1),M(2),C(2)]),('M0-3',[M(0),M(1),M(2),M(3)]),('M1M9',[M(1),M(9)])]:\n    print(name, 'quad', kern_cv_blocks(bl,q,[1e-3,1e-2]), 'exp', kern_cv_blocks(bl,e,[1e-3,1e-2]))\n"
}
```

