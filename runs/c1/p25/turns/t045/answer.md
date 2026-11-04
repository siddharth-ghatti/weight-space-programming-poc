## reasoning summary

I'm working out the start model's honest readout: the (2,6,10) combination plus an approximate C11 term, where the final CLS is about 2·M11 plus its own part. M11 is the mean hidden state at index 11, which is the input to layer 11 and already includes the edits in the base reads. The previous editor's CV for the start model was about 97.8, and I'm checking whether (2,6,10,11) can be treated as linear.

I'm weighing a kernel ridge approach, with a kernel built from the cosine-style term on z-scored, L2-normalized features or an RBF alternative. The note's ~98.0 result used a kernel on z-scored M9 with no L2 plus a linear kernel on M2, so I want to reproduce that combination and write a function that sums an RBF kernel on M9 with a scaled linear kernel on z(M2).



Linear readouts top out near 97.5% even at depth 9, so I'll re-derive the kernel numbers for depths 7–10 and add the start model's readout as a reference.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom cvlib import folds\nd=np.load('base_reads.npz'); y=d['y']; R=np.load('reserved.npy')\nkeep=np.setdiff1d(np.arange(768),R)\nM={j:d[f'mean_hidden_L{j}'][:,keep] for j in range(13)}\ndef z(X): return (X-X.mean(0))/(X.std(0)+1e-6)\ndef kcv(K, y, lams, seeds=(0,1,2)):\n    Y=np.eye(8)[y]; out={}\n    for lam in lams:\n        a=[]\n        for s in seeds:\n            f=folds(y,5,s); S=np.zeros((len(y),8))\n            for i in range(5):\n                tr=f!=i; te=f==i\n                Ktr=K[np.ix_(tr,tr)]; Yc=Y[tr]-Y[tr].mean(0)\n                A=np.linalg.solve(Ktr+lam*np.eye(tr.sum()), Yc)\n                S[te]=K[np.ix_(te,tr)]@A\n            a.append((S.argmax(1)==y).mean())\n        out[lam]=round(100*np.mean(a),2)\n    return out\ndef quad(X):\n    Z=z(X); Z/=np.linalg.norm(Z,axis=1,keepdims=True); Cc=Z@Z.T; return Cc+0.5*Cc**2\ndef rbf(X,g):\n    Z=z(X)/np.sqrt(X.shape[1]); sq=(Z**2).sum(1); D=sq[:,None]+sq[None]-2*Z@Z.T; return np.exp(-g*D)\ndef lin(X): Z=z(X)/np.sqrt(X.shape[1]); return Z@Z.T\nfor j in [7,8,9,10]:\n    for name,K in [('quad',quad(M[j])),('quad+linM2',quad(M[j])+lin(M[2])),('rbf.35+linM2',rbf(M[j],.35)+lin(M[2])),('rbf.35+linM2+linM6',rbf(M[j],.35)+lin(M[2])+lin(M[6]))]:\n        print(j,name,kcv(K,y,(3e-3,1e-2,3e-2)))"
}
```

