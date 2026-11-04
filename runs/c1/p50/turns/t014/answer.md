## reasoning summary

The block-normalized features land a bit below the unnormalized run: layers 0–2 give 97.1% here, against 97.4% for layers 0+1 before. Adding the original features with an RBF kernel reaches 98.06%. Early-layer features seem to plateau near 97%, so getting to 98% looks like it needs deeper features. That leaves a trade-off between keeping many layers, which costs more, and finding a shallower design that works just as well. I'm starting to weigh the costs, beginning with what dropping the FFN entirely would save.

I'm weighing the parameter budget of different designs, like dropping the FFN from the layers, or keeping only a few layers. A five-layer model with medium-sized FFNs lands around 40M, which is a plausible target, and I'm considering whether skipping middle layers would hold up.

I'm weighing whether full-depth models with narrower FFNs keep more of the information than a few wide layers. Twelve layers at m=256 comes to about 57.6M, which isn't clearly better than four full layers at 52.8M. Per-layer cost is dominated by attention, so narrowing the FFN only helps so much. I'm also wondering whether skipping some layers, say keeping the early prefix through layer 5, could approximate deeper features more cheaply.

I'm wondering whether the test set is simply easier than my cross-validation suggested, or whether training on the full data just fits better than an 80% split. To check, I'd compare cross-validation with 10 folds against 5 folds to see how much accuracy rises with more training data. I'm also planning to try RBF on a few other layer combinations, like (0,3,6,9,12), (6,12), and (2,6,10) without layer 12, to see which set carries the deeper information.

The 5-fold estimate probably understates what training on the full set would give, though test noise at this size is large enough that the original's 98.31% could partly be luck. To be reasonably safe I'd want cross-validated accuracy of at least 98%, which means the features need to match or beat the original with RBF at 98.06%. That points toward needing deeper features.

I'm considering cheaper routes to deep-quality features: dropping the FFNs in later layers, or keeping only a prefix like layers 0 through 6. The aim is to find a structure that reaches about 98% without the full 12-layer cost.

I'm now setting up RBF cross-validation on prefix candidates, pairing the mean features for layers 0 through j with the CLS features at each depth, for j from 4 up to 8.

I'm weighing whether a shallower design can match the original's features. The idea is to keep the carried means from the early layers and add a CLS summary from a deeper layer, with one head changed to uniform attention in the last kept layer. I want to test RBF cross-validation on those prefix combinations across depths 3 through 12.



Prefix-only features plateau near 97.1–97.4% even with an RBF head, while deep features from layers 6 to 12 reach about 98%. I'm next mapping RBF CV against prefix depth to chart the accuracy-versus-depth tradeoff.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\nd=load_base(); y=labels()\nM=lambda l: d[f'mean_hidden_L{l}'][:,KEEP]; Cc=lambda l: d[f'cls_hidden_L{l}'][:,KEEP]\ndef krr_cv(K, y, lams, k=5, seeds=(0,1)):\n    out={}; Y=np.eye(8)[y]\n    for lam in lams:\n        accs=[]\n        for seed in seeds:\n            f=folds(y,k,seed); S=np.zeros((len(y),8))\n            for i in range(k):\n                tr=f!=i; te=f==i\n                A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0)); S[te]=K[np.ix_(te,tr)]@A\n            accs.append((S.argmax(1)==y).mean())\n        out[lam]=round(float(np.mean(accs)),4)\n    return out\ndef rbf(X,gf=1):\n    Xs=X-X.mean(0); sq=(Xs**2).sum(1); D2=sq[:,None]+sq[None,:]-2*Xs@Xs.T\n    return np.exp(-D2/(gf*np.median(D2)))\ndef blockn(Xs): return np.concatenate([(x-x.mean(0))/np.sqrt(((x-x.mean(0))**2).sum(1).mean()) for x in Xs],1)\nfor j in [3,4,5,6,7,8,9,10,12]:\n    sel=sorted(set([0, j//2, j]))\n    r1=krr_cv(rbf(blockn([M(l) for l in sel]+[Cc(j)])),y,(0.003,0.01))\n    r2=krr_cv(rbf(blockn([M(j),Cc(j)])),y,(0.003,0.01))\n    r3=cv_acc(blockn([M(l) for l in sel]+[Cc(j)]),y,lams=(0.03,0.1,0.3))\n    print(j, sel,'rbf means+cls',r1,'| rbf Mj+Cj',r2,'| lin', {k:round(v,4) for k,v in r3.items()})"
}
```

