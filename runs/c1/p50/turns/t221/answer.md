## reasoning summary

I'm comparing the 2-stage pipeline against the joint fit, which comes out about 0.25% lower. Even the D0 variant loses 0.12, and I'm trying to work out why a refit on the new carry direction would lose anything at all. My guess is that the Lin8 scores, fitted in-sample on the training fold, are overfit there.

I'm weighing whether the refit is the problem: the 8-dim block with no penalty would put a large weight on Lin8, which looks more predictive on training data than it really is. So instead of refitting B, I'm considering deploying the joint fit directly, with the Lin8 scores scaled by k and the bias kept, and I'll test that variant with and without the noise term. I also suspect the earlier CV estimates for the previous pool may have been optimistic by about 0.25%.

I'm now considering whether a cross-fitted version of the Lin8 scores could fix the overfit in the refit, but the notes already say stacked out-of-fold scores didn't help while the joint fit did. So the more promising route is to deploy the joint fit's exact linear block, with the carry updated and the pool set to match, rather than refitting.

I'm settling on the joint-fit kernel coefficients with B set to identity, scaled by 1/(a·mean k), and I still need to check whether the per-sample k noise at 15% cv hurts, so I'll test with and without it. Updating the carry means editing layer 0 head 0's value dims 56–63 and the output dense into the residual stream, with the value's mean over tokens giving the weight and bias terms.

I'm working out how to change the carry direction in layer 0: replacing the value weights for dims 56–63 with the new direction mapped back to raw coordinates, and scaling it so the carried score variance matches the old one. Next I'm checking whether that change also feeds into the downstream content layers, since the residual-stream carry is present at every token.

I'm checking whether the change in carry direction leaves the LayerNorm statistics in later layers essentially unchanged, since the carried variance per token is small compared to the token variance. The remaining question is whether the per-sample scale k_i shifts much, and the verification pass will confirm that against the actual logits.

I'm planning the verification sequence: install the new carry and pool, run a read pass over the 1024 ids to get logits and last-layer features, and check those against the numpy predictions. The remaining 472 ids get a second read pass and a final check, leaving three passes in reserve.



A two-stage pipeline loses about 0.25% to the joint fit, which I attribute to stacking overfit, so earlier CV numbers were likely optimistic. I'm now simulating a D_exact variant that deploys the joint fit directly, with and without the k_i scale noise.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nimport numpy as np, json, time\nfrom lib7 import *\nallids=[r['id'] for r in practice()]; y=labels_for(allids, practice, LABELS); Y=np.eye(8)[y]\nf=np.load('feats_drop46.npz'); M,C,M0=f['M'],f['C'],f['M0']\nk=np.load('carry_k.npy')\ndef prep(M,C,tr,c=10,na=1008):\n    mu=M[tr].mean(0); Z=pool_z(M,C,c,mu); Zt=Z[tr]; Wf=anchor_dirs(Zt); Kc=gelu(Zt@Wf.T); Kc=.5*(Kc+Kc.T)\n    piv=pivchol(Kc,na); return feats(Z,Wf[piv]), Kc[np.ix_(piv,piv)]\ndef run(seed, kf=10, lam=.05, c0=.5, c=10):\n    fo_=folds(y,kf,seed); P={n:np.zeros(1496,int) for n in ['A','Dk','D1','Dk_kmean']}\n    for i in range(kf):\n        tr=fo_!=i; te=~tr; Lin=lin_feats(M0,tr)\n        Pn,Kn=prep(M,C,tr,c=c)\n        cn,bn=fit(Pn[tr],Lin[tr],Y[tr],Kn,lam,c0)\n        Sk=Pn@cn[:1008]; Sl=Lin@cn[1008:]\n        P['A'][te]=(Sk+Sl+bn)[te].argmax(1)\n        P['Dk'][te]=(Sk+k[:,None]*Sl+bn)[te].argmax(1)\n        P['D1'][te]=(Sk+Sl+bn)[te].argmax(1)\n        # pool knows per-sample nothing; but rescale lin by 1/mean(k on train)\n        P['Dk_kmean'][te]=(Sk+(k/k[tr].mean())[:,None]*Sl+bn)[te].argmax(1)\n    return {n:(p==y).mean() for n,p in P.items()}\nR=[run(s) for s in range(10,16)]\nfor n in R[0]: print(n, np.round([r[n] for r in R],4), round(np.mean([r[n] for r in R]),4))"
}
```

