## reasoning summary

I'm seeing that the RBF kernel gives a small gain, with the L1 mean reaching about 97.26%, and that smaller lambdas might do better. I want to widen the search to lam 0.001 and try the kernel on deeper layer combinations, including L12. Implementing a true RBF kernel inside the model isn't feasible, so the pooler's 768 tanh features would only approximate it, and using the final-layer FFN at CLS is an option, though its m units are shared and costly.

I'm weighing whether 768 random tanh features of the CLS state could match the RBF kernel, since I can't build one directly. The pooler is a fixed projection, and the classifier on top would be a closed-form ridge fit, so it's essentially a 768-unit random-feature model. I'm also considering whether Nyström-style landmarks could work, but for normalized features the RBF reduces to a dot-product function, and tanh only approximates that.

I'm weighing a Nyström-style alternative: pick training points as centers and use tanh of their dot products with the input as features, then fit ridge on top, something like an extreme learning machine with data-chosen centers. I want to test whether 768 such features on the layer-1 representation can get close to the RBF kernel's accuracy. I'm also noting that the final layer's FFN at CLS would add only a little if its width is small.

I'm weighing whether a reduced readout, with the carry limited to M(1), can hold up against the wider combinations. The kernel on M(1) gives 97.26%, against 97.46% for M(0,1,2), so the extra layers buy only a small gain. Next I want kernel CV across L1 through L12, both individually and in combinations, with smaller lambdas.



An RBF kernel lifts layer-1 CV accuracy from 95.9% to 97.3%, so a nonlinear readout matters; I'm now checking the kernel result per layer with smaller penalties.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0, str(WORK))\nfrom cvlib import *\nR = np.load(WORK/'base_reads.npz'); y=R['y']; res=np.load(WORK/'reserved.npy')\nkeep = np.setdiff1d(np.arange(768), res)\nM = lambda L: R[f'mean_hidden_L{L}'][:,keep]\ndef krr_cv(X, y, gammas, lams, k=5, seed=0):\n    f=folds(y,k,seed); out={}\n    mu=X.mean(0); sd=X.std(0)+1e-6; Z=(X-mu)/sd\n    sq=(Z**2).sum(1); D2 = sq[:,None]+sq[None,:]-2*Z@Z.T; D2/= Z.shape[1]\n    Y=np.eye(8)[y]\n    for g in gammas:\n        K=np.exp(-g*D2)\n        for lam in lams:\n            S=np.zeros((len(y),8))\n            for i in range(k):\n                tr=f!=i; te=f==i\n                Ytr=Y[tr]-Y[tr].mean(0)\n                A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()), Ytr)\n                S[te]=K[np.ix_(te,tr)]@A\n            out[(g,lam)]=round(float((S.argmax(1)==y).mean()),4)\n    return out\nfor L in range(0,13):\n    r=krr_cv(M(L), y, gammas=[0.3,1.0], lams=[0.001,0.01])\n    print(L, r, 'best', max(r.values()))\n"
}
```

