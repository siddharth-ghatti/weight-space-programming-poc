## reasoning summary

Using 768 centers, the kernel approximation drops accuracy only slightly, from about 97.9% at L9 to around 96.7–97.0% at L0 and L1. Even the full kernel ceiling for the early layers sits near 97.3%, below the original model's 97.8%, so that looks risky. I'm now going to step back and reconsider the overall approach.

The minimal configuration is about 26.8M parameters, but any setup using deep layers costs at least 45.7M even with no FFN. So the middle ground is shallow layers, around k=2 to 4. I'm wondering whether shallow layers can carry more signal, perhaps by splitting the pooling heads so some capture position-specific features like sentence openings.

I'm weighing whether to pool M1 and C1 from layer 0 together in a k=2 model, since M0 alone can't be pooled separately without repurposing a head. I'll check how those feature combinations score in cross-validation.

I'm checking whether shallow models can reach the 97.8% mark at all, since even full kernel ridge on the early-layer features tops out near 97.3%. The original's cross-validation was 97.8%, and my ridge variants with deep layers landed at 97.7–98.0%, so the test set looks about as hard as CV or slightly easier. That makes me want CV of at least 97.8%, which points toward a deeper k=10 kernel at around 98.3% CV, though it costs roughly 48M parameters even with no FFN.

Now I'm weighing whether a deeper kernel model, around k=10, is worth its extra parameters, against shallow layers combined with the pooled M0 features.

I'm weighing whether a deeper kernel is worth its parameters, and asking whether attention-only layers without FFN would hold useful signal. I'm also considering what deeper layers capture that shallow ones miss, like negation scope or the difference between claims procedure and conditions, and I want to see which classes get confused.

I'm comparing where the L1 and L9 kernel models get confused, looking for one specific class error that position-specific pooling might fix.



With only 768 centers, the kernel drops about 0.3–0.5 points, and even the full shallow kernel stays under 97.3%. Next I'll check which classes get confused at shallow versus deep layers and test cheap extra features for the 2-layer model.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nexec(open(WORK/'lib.py').read())\nF = np.load(WORK/'base_feats.npz'); y=F['y']\nM = {L: F[f'mean_hidden_L{L}'][:,KEEP] for L in range(13)}\nC = {L: F[f'cls_hidden_L{L}'][:,KEEP] for L in range(13)}\ndef zs(X,tr): mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; return (X-mu)/sd/np.sqrt(X.shape[1])\ndef krr_scores(blocks, y, gamma, lam, k=5, seed=0, weights=None):\n    n=len(y); S=np.zeros((n,8))\n    for te in folds(n,k,seed,y):\n        tr=np.setdiff1d(np.arange(n),te)\n        Zs=[zs(X,tr)*(1 if weights is None else w) for X,w in zip(blocks, weights or [1]*len(blocks))]\n        Z=np.concatenate(Zs,1)/np.sqrt(len(blocks)); sq=(Z**2).sum(1)\n        K=np.exp(-gamma*np.maximum(sq[:,None]+sq[None]-2*Z@Z.T,0))\n        T=np.eye(8)[y[tr]]-1/8\n        a=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(len(tr)),T); S[te]=K[np.ix_(te,tr)]@a\n    return S\ndef acc(S): return round((S.argmax(1)==y).mean(),4)\nfor name,bl in [('M1',[M[1]]),('M0+M1',[M[0],M[1]]),('M1+C1',[M[1],C[1]]),('M0+M1+C1',[M[0],M[1],C[1]]),('M2+C2',[M[2],C[2]]),('M0+M9',[M[0],M[9]]),('M6',[M[6]]),('M0+M6',[M[0],M[6]])]:\n    print(name,[acc(krr_scores(bl,y,g,0.01)) for g in (0.2,0.4,0.8)])\nS1=krr_scores([M[1]],y,0.4,0.01); S9=krr_scores([M[9]],y,0.4,0.01)\ndef conf(S):\n    cm=np.zeros((8,8),int)\n    for t,p in zip(y,S.argmax(1)): cm[t,p]+=1\n    return cm\nprint(conf(S1)); print(conf(S9))\n"
}
```

