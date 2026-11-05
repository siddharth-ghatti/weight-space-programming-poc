## reasoning summary

I'm working out the token budget: about 21.5k tokens per layer for layers 0 through 4, and 2,000 CLS samples for layer 5. Next I want to test the linear-unit idea offline on a holdout split by sequence, holding out roughly 20% of them. For widths of 1024, 1280, and 1536, I'll compare the baseline of keeping the top-variance units with ridge compensation, measuring relative error of the FFN output on the holdout.

I'm now weighing a reduced-rank version of the linear units: fit a joint ridge on the features, then keep only r directions from the coefficient matrix on the activations. I'm considering how to choose those directions, either from the SVD of the coefficient matrix or from the projected linear contribution, so the linear part stays within rank r.

I'm checking the rank-r reduction of the linear part: taking the top right singular directions of the linear contribution, so each new feature is the activations projected onto C·V_i. Then I'd refit jointly with those features. Before running anything I want to confirm which regularization value the m=1536 build actually used, since the holdout baseline depends on it.

I'm about to write the experiment itself, so I'm checking which regularization value the m=1536 build used before running the holdout comparison.



About 21.5k token samples per layer are collected for L0–4 and 2,000 for L5. Next I'll compare the output error of the current method against a version that swaps r units for linear units, on a held-out 20% of sequences.



## tool_use: run_python

```json
{
 "code": "exec(open('seq.py').read())\nimport time\nrng=np.random.default_rng(0)\ndef experiment(L, m, rs=(0,64,128,256), lam_rel=0.05):\n    A, S = all_data(L); A=A.astype(np.float64)\n    useq=np.unique(S); te_seq=rng.choice(useq, len(useq)//5, replace=False)\n    te=np.isin(S, te_seq); tr=~te\n    pre=f\"roberta.encoder.layer.{L}.\"\n    W1=base(pre+\"intermediate.dense.weight\").astype(np.float64); b1=base(pre+\"intermediate.dense.bias\")\n    W2=base(pre+\"output.dense.weight\").astype(np.float64)\n    H=gelu(A@W1.T+b1); F=H@W2.T\n    sc=H[tr].var(0)*(W2**2).sum(0); order=np.argsort(-sc)\n    Fte=F[te]; den=((Fte-Fte.mean(0))**2).sum()\n    out={}\n    for r in rs:\n        keep=np.sort(order[:m-r]); drop=np.setdiff1d(np.arange(3072),keep)\n        T=H[:,drop]@W2[:,drop].T\n        Xk=H[:,keep]\n        if r>0:\n            # linear directions: ridge of T on [Xk, A], then reduced rank of linear part\n            Z=np.c_[Xk, A]\n            mu=Z[tr].mean(0); Zc=Z[tr]-mu; G=Zc.T@Zc; lam=lam_rel*np.trace(G)/Z.shape[1]\n            Bz=np.linalg.solve(G+lam*np.eye(Z.shape[1]), Zc.T@(T[tr]-T[tr].mean(0)))\n            C=Bz[len(keep):]  # 768x768\n            P=(A[tr]-A[tr].mean(0))@C\n            _,_,Vt=np.linalg.svd(P, full_matrices=False)\n            D=C@Vt[:r].T   # input dirs 768 x r\n            Xk=np.c_[Xk, A@D]\n        mu=Xk[tr].mean(0); Xc=Xk[tr]-mu; G=Xc.T@Xc; lam=lam_rel*np.trace(G)/Xk.shape[1]\n        B=np.linalg.solve(G+lam*np.eye(Xk.shape[1]), Xc.T@(T[tr]-T[tr].mean(0)))\n        pred=(Xk[te]-mu)@B + T[tr].mean(0)\n        err=((T[te]-pred)**2).sum()/den\n        err0=((T[te]-T[tr].mean(0))**2).sum()/den\n        out[r]=(round(err,4), round(err0,4))\n    return out\nt=time.time()\nfor L in [1,3]:\n    print(L, experiment(L,1024), time.time()-t)\n"
}
```

