## reasoning summary

I'm getting roughly 96.5% with 8 ridge directions, against a 95.6-95.9% baseline, while concatenating the full h0 reached 97.4. The 8-direction result is still about a point short of that, and I suspect stacking overfit: the ridge directions are fit on the training fold, so they may look overconfident on that same data.

I'm wondering whether the gap to 97.4 comes from stacking overfit, since the ridge directions were fit on the same training fold and may be overconfident there. A cleaner route might be to take the m0 weights from a joint ridge fit and refit a second-stage ridge on those carried directions within the fold, then compare that against the joint model directly.



Layer-0 ridge directions alone reach about 96.5% (mean-only 95.7%, full concatenation 97.4%), so stacking loses some gain. Next I'll try taking the 8 directions from the layer-0 block of the joint ridge on [mean h11, mean h0].



## tool_use: run_python

```json
{
 "code": "exec(open('helpers.py').read())\nlab = labels_all(); y = lab[:1024]; folds = make_folds(1024)\nm11 = load('mean',11); m0 = load('mean',0)\nJ = np.concatenate([m11,m0],1)\nfor lamj in [0.2,0.4,0.8]:\n    out=[]\n    cj=0\n    for lam in (0.1,0.2,0.4,0.8):\n        c=0\n        for f in folds:\n            tr=np.setdiff1d(np.arange(1024),f)\n            Wf,b = ridge_fit(J[tr],y[tr],lamj)\n            if lam==0.1: cj += ((J[f]@Wf+b).argmax(1)==y[f]).sum()\n            A = Wf[768:]\n            X = np.concatenate([m11, m0@A],1)\n            W2,b2 = ridge_fit(X[tr],y[tr],lam)\n            c+=((X[f]@W2+b2).argmax(1)==y[f]).sum()\n        out.append(round(c/1024,4))\n    print('lamj',lamj,'joint',round(cj/1024,4),'stack',out)\n"
}
```

