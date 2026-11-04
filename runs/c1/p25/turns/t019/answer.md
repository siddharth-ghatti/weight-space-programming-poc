## reasoning summary

Linear models plateau around 96%, while the kernel reaches 97.2% on M1, so the nonlinearity clearly helps. The small regularization suggests the kernel is acting as a local similarity measure, which fits a template-like structure in the data. I'll check whether kNN with cosine similarity on M1 gets close to that 97%.

I'm weighing whether a GELU layer with a strongly negative bias could act as a sharp match detector, similar to an RBF unit, though it would need many centers. A tanh readout with a very negative bias looks like a possible alternative, with its own saturation issues.

I'm checking whether a tanh unit with a strongly negative bias behaves like an exponential kernel in its tail, since tanh(z)+1 is roughly 2e^{2z} there. The catch is float32 precision near -1, which limits the usable range of the pre-activation.

I'm sketching a Nyström-style RBF approximation using 768 centers, sampled class-balanced, to match the pooler width. I'll test both the plain exponential features and the tanh variant to see which holds up numerically, with the classifier bias absorbing the constant offset.

I'm running the Nyström RBF check now, using 768 class-balanced centers to match the pooler width.



Linear readouts plateau near 96%, so the kernel's extra ~1.2% likely comes from nonlinear similarity matching. I'm next testing kNN and a Nyström RBF readout with 768 centres.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0, str(WORK))\nfrom cvlib import *\nR = np.load(WORK/'base_reads.npz'); y=R['y']; res=np.load(WORK/'reserved.npy')\nkeep = np.setdiff1d(np.arange(768), res)\nM = lambda L: R[f'mean_hidden_L{L}'][:,keep]\nY=np.eye(8)[y]\nrng=np.random.RandomState(0)\ndef nys_cv(X, gamma, lams, ncent=768, seeds=(0,1), mode='rbf', lin=False):\n    out={l:[] for l in lams}\n    for s in seeds:\n        f=folds(y,5,s); S={l:np.zeros((len(y),8)) for l in lams}\n        for i in range(5):\n            tr=np.where(f!=i)[0]; te=np.where(f==i)[0]\n            mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6\n            Z=(X-mu)/sd; d=Z.shape[1]\n            Z=Z/np.linalg.norm(Z,axis=1,keepdims=True)  # unit-normalize (mimic LN)\n            cidx=[]\n            for c in range(8):\n                ic=tr[y[tr]==c]; cidx+=list(rng.choice(ic, min(ncent//8,len(ic)), replace=False))\n            C=Z[cidx]\n            F=np.exp(gamma*(Z@C.T-1))\n            if lin: F=np.concatenate([F, Z],1)\n            Ftr=F[tr]; fm=Ftr.mean(0); Ftr=Ftr-fm; Fte=F[te]-fm\n            G=Ftr.T@Ftr; B=Ftr.T@(Y[tr]-Y[tr].mean(0))\n            for l in lams:\n                S[l][te]=Fte@np.linalg.solve(G+l*np.eye(G.shape[0]),B)\n        for l in lams: out[l].append((S[l].argmax(1)==y).mean())\n    return {l:round(float(np.mean(v)),4) for l,v in out.items()}\n# knn cosine\nX=M(1); f=folds(y,5,0); Z=(X-X.mean(0))/(X.std(0)+1e-6); Z/=np.linalg.norm(Z,axis=1,keepdims=True)\nSim=Z@Z.T\nfor kk in [1,3,5]:\n    acc=[]\n    pred=np.zeros(len(y),int)\n    for i in range(5):\n        tr=np.where(f!=i)[0]; te=np.where(f==i)[0]\n        Ss=Sim[np.ix_(te,tr)]; nn=np.argsort(-Ss,1)[:,:kk]\n        votes=np.zeros((len(te),8))\n        for j in range(kk): votes[np.arange(len(te)), y[tr][nn[:,j]]]+=1\n        pred[te]=votes.argmax(1)\n    print('knn',kk,(pred==y).mean())\nfor g in [5,10,20]:\n    print('nys rbf', g, nys_cv(X, g, [1e-3,1e-2,1e-1]))\nprint('nys full 1196', nys_cv(X, 10, [1e-3,1e-2], ncent=1196))\n"
}
```

