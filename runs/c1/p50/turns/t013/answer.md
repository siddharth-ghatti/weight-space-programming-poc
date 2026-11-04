## reasoning summary

Errors seem to be concentrated in the semantically hard cases, and the deeper features appear to help there. I want to test which layer structures work: I suspect dropping middle layers in BERT is fairly harmless, so I'll try keeping subsets and check how good their mean-pooled features are with a few quick forward passes.

I'm checking whether the first layers could carry most of the signal: I'd compare RBF heads on combinations like (0,1,2), (0,2), and (0,6), plus a CLS feature from layer 12. Compressing each layer's mean with PCA might let several layers fit in the feature budget, so I'll need cross-validation to see whether that holds up.

Carrying the per-layer means into the final CLS through reserved dimensions would mean isolating those dimensions in every layer, which at 256 dims per layer would cut too deeply into the model's 768-dim capacity. The residual stream of the last pooling input already holds earlier information, but the (0,6) linear result suggests it isn't preserved linearly, so a nonlinear head might still recover more of it.

I'm considering whether a head whose query responds only to the CLS position could carry earlier-layer information without disturbing other tokens' attention to CLS. Since the CLS embedding is distinct at the first layer, a query direction that is large for CLS and near zero for everything else might work approximately.

I'm sketching out a few candidate designs and weighing their costs, with a budget of about 15 more experiments. The first one collapses layer 0 into a pooling layer with a random-feature tanh head, which I expect lands around 96.5–96.9% but probably isn't accurate enough. I'm about to look at a second option that keeps layer 0 intact.

I'm weighing a second design where layer 1 becomes the pooling layer over h1, since h1 already carries h0 information through the residual stream. Its parameter count depends on the FFN width, so I'm comparing a full m=3072 variant against narrower ones. A third option carries means from layers 0 through 2 into a nonlinear head, and I'm wondering whether attention-based pooling could improve on the RBF bag-of-embeddings result.

Attention-based pooling with several query patterns might add useful signal beyond the RBF bag-of-embeddings result. I'm not sure it will pay off, so I want to check it against the base features offline first, comparing RBF fits on different layer combinations and whether PCA compression keeps the accuracy.



Many of the errors are genuinely hard semantic cases, and the deeper features fix about a third of the bag-of-words ones. Next I'm checking which cheap prefix-layer combinations reach roughly 98% CV with an RBF head, to see how many layers I need to keep.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\nd=load_base(); y=labels()\nM=lambda l: d[f'mean_hidden_L{l}'][:,KEEP]; Cc=lambda l: d[f'cls_hidden_L{l}'][:,KEEP]\ndef krr_cv(K, y, lams, k=5, seeds=(0,1)):\n    out={}; Y=np.eye(8)[y]\n    for lam in lams:\n        accs=[]\n        for seed in seeds:\n            f=folds(y,k,seed); S=np.zeros((len(y),8))\n            for i in range(k):\n                tr=f!=i; te=f==i\n                A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0)); S[te]=K[np.ix_(te,tr)]@A\n            accs.append((S.argmax(1)==y).mean())\n        out[lam]=round(float(np.mean(accs)),4)\n    return out\ndef rbf(X,gf=1):\n    Xs=X-X.mean(0); Xs=Xs/np.sqrt((Xs**2).sum(1).mean()); sq=(Xs**2).sum(1); D2=sq[:,None]+sq[None,:]-2*Xs@Xs.T\n    return np.exp(-D2/(gf*np.median(D2)))\ndef blockn(Xs): return np.concatenate([(x-x.mean(0))/np.sqrt(((x-x.mean(0))**2).sum(1).mean()) for x in Xs],1)\nfor name,blocks in [('L1',[M(1)]),('L2',[M(2)]),('0,2',[M(0),M(2)]),('0,1,2',[M(0),M(1),M(2)]),('0,3',[M(0),M(3)]),('0,1,2,3',[M(0),M(1),M(2),M(3)]),('0,6',[M(0),M(6)]),('0,6,12',[M(0),M(6),M(12)]),('2,6,10,c12',[M(2),M(6),M(10),Cc(12)]),('L1+c2',[M(1),Cc(2)])]:\n    X=blockn(blocks); print(name, krr_cv(rbf(X,1),y,(0.003,0.01,0.03)))"
}
```

