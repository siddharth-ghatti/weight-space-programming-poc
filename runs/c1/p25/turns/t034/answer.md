## reasoning summary

Dropping layers costs some quality here: the edited model at L5 lands near the base model's L5/L6 score, and at L7 it gets close to base L7 but still trails the base model's L9 result.

I'm weighing whether a readout that carries more than the current 8 dimensions per channel could push accuracy up at a fixed depth. Something like a kernel over several layers jointly, rather than just carrying linear class scores forward, might recover some of the gap to the full-depth model.

Splitting the readout into a richer multi-dimensional carry would mean reserving far more of the residual stream, which might hurt BERT, though the earlier cut of 8 dims seemed fine. Shallow layers don't seem to beat the mean-pooled kernel, and the useful contextual signal shows up only around layer 8 and later. So cleverer pooling on a shallow model probably won't recover that deep-layer semantics.

I'm checking whether shallow features with smarter pooling could match the deep model. The gains from deep layers might come from specific cases like negation or question-answer structure. So I'll compare which sentences the layer-0 kernel gets wrong against the deep model's predictions, looking especially at FAQ pairs where the question names another class.



Dropping layers costs about as much accuracy as it saves, so full depth is what buys quality, landing near 88–95M. Before committing, I'm comparing out-of-fold predictions to see whether the sentences deep features fix share a pattern that cheap pooling could capture.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0, str(WORK))\nfrom cvlib import *\nR = np.load(WORK/'base_reads.npz'); y=R['y']; res=np.load(WORK/'reserved.npy'); ids=R['ids']\nkeep = np.setdiff1d(np.arange(768), res)\nM = lambda L: R[f'mean_hidden_L{L}'][:,keep]\nY=np.eye(8)[y]; P={x['id']:x for x in practice()}\ndef nz(X):\n    Z=(X-X.mean(0))/(X.std(0)+1e-6); return Z/np.linalg.norm(Z,axis=1,keepdims=True)\ndef zsc(X): return (X-X.mean(0))/(X.std(0)+1e-6)/np.sqrt(X.shape[1])\ndef oof(K, lam=1e-2, s=0):\n    f=folds(y,5,s); S=np.zeros((len(y),8))\n    for i in range(5):\n        tr=f!=i; te=f==i\n        A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0)); S[te]=K[np.ix_(te,tr)]@A\n    return S\nq=lambda C: C+0.5*C**2\nZ0=nz(M(0)); Z9=nz(M(9)); L1=zsc(M(1))\nS0=oof(q(Z0@Z0.T)); S9=oof(q(Z9@Z9.T)+L1@L1.T)\np0=S0.argmax(1); p9=S9.argmax(1)\nprint('acc', (p0==y).mean(), (p9==y).mean())\nfixed=np.where((p0!=y)&(p9==y))[0]; broke=np.where((p0==y)&(p9!=y))[0]\nprint('fixed by deep', len(fixed), 'broken', len(broke))\nimport collections\nprint('registers fixed', collections.Counter(P[ids[i]]['register'] for i in fixed))\nprint('registers of M0 errors', collections.Counter(P[ids[i]]['register'] for i in np.where(p0!=y)[0]))\nprint('registers overall', collections.Counter(x['register'] for x in practice()))\nfor i in fixed[:25]:\n    x=P[ids[i]]; print(f\"[{LABELS[y[i]][:8]} M0->{LABELS[p0[i]][:8]}] ({x['register'][:8]}) {x['text'][:150]}\")\n"
}
```

