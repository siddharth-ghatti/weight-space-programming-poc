## reasoning summary

The network now agrees with the numpy reference to about 2e-4 on the logits, and training accuracy is 100%. Out-of-fold accuracy suggests roughly 97.8% on test. I'm saving notes, then weighing whether the remaining budget of 39 passes and 270 runs is worth spending on further improvements, such as tuning lam, which carry layers to use, and the aggregator's c, along with adding more carry heads.

I'm weighing a different classifier on the same features, such as shrinkage LDA or ridge with per-block regularization. I'm also considering random tanh features in the pooler fit by ridge, though that may just overfit.

I'm thinking about whether a random tanh pooler with a ridge-fit classifier could help, though with about 1500 samples I doubt the gain is large. Before touching the network, I want to check kernel ridge with an RBF kernel on the joint features in numpy, as an upper bound for what nonlinear classifiers could add. If that comes out clearly above 97.8%, the random-pooler idea would be worth pursuing; LDA variants are also on the list.



The network now matches the numpy reference within 2e-4 on the logits, and the honest generalization estimate is about 97.8% from CV. I'm updating the notes, then testing whether shrinkage LDA or kernel ridge adds anything.



## tool_use: run_python

```json
{
 "code": "with open(WORK/'notes.md','a') as f:\n    f.write(\"\"\"- BUG found & fixed: R coords carry a common LN shift (up to ~2); decode weight 1/gw=10 saturated tanh -> 51% acc. Fix: zero-sum decode 10*(I-11^T/8) on R block.\n- v3 VERIFIED (pass_11): in-sample acc 1.0, logits match numpy expectation within 2e-4. CURRENT SAVED EDITS = v3 (good, submit-able). OOF est ~97.8%.\n\"\"\")\nexec(open(WORK/'build2.py').read())\nlab=np.load(WORK/'labels.npy'); N=len(lab); folds=np.load(WORK/'folds.npy'); Y=np.eye(8)[lab]\nF=dict(np.load(WORK/'prep_feats.npz')); A=np.load(WORK/'v2_layer_scales.npy'); abar=A.mean(0)\nXs=np.concatenate([F[p][:,KEEP]*(A[:,k]/abar[k])[:,None] for k,p in enumerate(['m2','m6','m10'])]+[F['c12'][:,KEEP]],1)\nacc=lambda o:round((o.argmax(1)==lab).mean(),4)\ndef cv(fitpred):\n    o=np.zeros((N,8))\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        mu=Xs[tr].mean(0); sd=Xs[tr].std(0)+1e-6\n        o[te]=fitpred((Xs[tr]-mu)/sd, lab[tr], (Xs[te]-mu)/sd)\n    return o\ndef lda(s):\n    def fp(Z,y,Ze):\n        mus=np.stack([Z[y==c].mean(0) for c in range(8)]); Rr=Z-mus[y]\n        # dual-ish: covariance d x d (3040) is ok\n        S=Rr.T@Rr/len(Z); S=(1-s)*S+s*np.trace(S)/S.shape[0]*np.eye(S.shape[0])\n        Wl=np.linalg.solve(S,mus.T); b=-0.5*np.sum(mus.T*Wl,0); return Ze@Wl+b\n    return fp\nfor s in [0.3,0.6,0.9]: print('lda',s,acc(cv(lda(s))),flush=True)\ndef krr(gamma,lam):\n    def fp(Z,y,Ze):\n        d=Z.shape[1]\n        sq=(Z**2).sum(1); K=np.exp(-gamma*(sq[:,None]+sq[None,:]-2*Z@Z.T)/d)\n        sqe=(Ze**2).sum(1); Ke=np.exp(-gamma*(sqe[:,None]+sq[None,:]-2*Ze@Z.T)/d)\n        Yt=np.eye(8)[y]; al=np.linalg.solve(K+lam*np.eye(len(K)),Yt-Yt.mean(0)); return Ke@al\n    return fp\nfor g in [0.1,0.3,1.0]:\n    for lam in [0.1,1.0]:\n        print('krr',g,lam,acc(cv(krr(g,lam))),flush=True)\n"
}
```

