## reasoning summary

I'm noting that the test set likely contains paraphrase-like near-duplicates, so kernel methods still look reasonable, with cross-validation as the honest estimate. Next I'm sketching Plan B, a kernel machine on the FFN in layer 11, and checking how the CLS representation gets normalized after LN1.

I'm now working out the feature construction for Plan B: using linear features on the standardized layer-11 CLS representation, with each unit's weight set to a training point's standardized direction scaled by sd, and a GELU applied after a scaling by the typical norm. I'm planning to fit the output weights and linear term jointly by ridge regression, choosing the scale, offset, and penalty by cross-validation. One caveat I'm noting is that if the centers are all the training points, the cross-validation has to pick centers only from each fold's training part.

I'm reasoning through how to realize the output as a residual: a sum of scaled directions with a large gain, plus a linear term built from paired GELU units. The unit budget works out to the training-point centers plus 16, which still fits within the 3072 intermediate width. Once the gain is large, the post-LayerNorm output should be close to a normalized version of that residual term.

I'm working out that the final vector is essentially the scaled residual r, since the large gain makes z negligible. The pooled outputs then become tanh of projections onto the direction vectors, which keeps the argmax intact. The one risk I'm flagging is that if the scores are small relative to z, that noise leaks into the prediction.

I'm checking the GELU choice next: the config uses the exact erf form, so I'll compute the features with scipy's erf and confirm later whether the tanh approximation would change anything. The open question is what cross-validation gives for this feature map, and I'll compute the fold features with centers taken only from the training part.

I'm now picking the grid for the activation's scale and offset, expecting cosine similarities between standardized points to sit mostly between 0 and 0.3. I'll also keep the RBF kernel ridge result of 98.2% as the benchmark, since an exact RBF can't be built from GELU alone.

I'm running cross-validation on the feature map next, with the kernel-style features built from the training-fold centers and the linear standardized features scaled by a factor, so I can tune how strongly the two parts are penalized relative to each other.



The data's near-duplicate paraphrases make a kernel-style classifier sensible, and I'm cross-validating its feature map in closed form next, with centres taken from the training fold only.



## tool_use: run_python

```json
{
 "code": "from scipy.special import erf\nlab=np.load('feats/y.npy'); folds=np.load('feats/folds.npy'); E=np.load('feats/edA_cls12.npy')\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\ndef run(a,t,lam,rho,useLin=True):\n    pr=np.zeros(len(lab),int)\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        mu=E[tr].mean(0); sd=E[tr].std(0)+1e-6\n        Z=(E-mu)/sd; R=np.sqrt(768)\n        C=Z[tr]/np.linalg.norm(Z[tr],axis=1,keepdims=True)\n        U=(Z@C.T)/R             # ~cosine-ish\n        F=gelu(a*U+t)\n        if useLin: F=np.concatenate([F, rho*Z/R],1)\n        Ftr=F[tr]; m=Ftr.mean(0)\n        Y=np.eye(8)[lab[tr]]-1/8\n        Wt=np.linalg.solve((Ftr-m).T@(Ftr-m)+lam*np.eye(F.shape[1]),(Ftr-m).T@Y)\n        pr[te]=((F[te]-m)@Wt).argmax(1)\n    return round((pr==lab).mean(),4)\nfor a in [2,5,10]:\n    for t in [-2,0,-1]:\n        print(a,t,[run(a,t,lam,1.0) for lam in [0.01,0.1,1]])\n"
}
```

