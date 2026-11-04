## reasoning summary

I'm comparing a few combinations of carried linear features from different layers, like M2 with M6 and a broader set from M1 through M7. I'm also noting that each carry head yields eight scores, and I'm starting to consider how two carry heads' linear features might combine into a kernel.

I'm working out how to combine the carried features from two heads, M2 and M5, into one joint ridge score. The idea is that each head writes its share into the same reserved dimensions, which then sum with the quadratic kernel part at readout. I'm leaning toward fixing those carry weights rather than refitting the classifier, so the carried dims aren't leaked from the training data.

The Nyström approximation means the fit has to happen in primal form on the feature maps that are actually implemented, so I'll have to fit against those. I also want to test more combinations for the M8 case, adding linear features from layers M1 through M7 alongside the quadratic term, aiming for around 97.9, and for M9 add more linear layers.



Adding linear terms from more layers raised the score to 97.75 (M8 quad + linM2 + linM5). Next I'll find which linear-carry layers work best at depths 8 and 9.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom cvlib import folds\nd=np.load('base_reads.npz'); y=d['y']; R=np.load('reserved.npy')\nkeep=np.setdiff1d(np.arange(768),R)\nM={j:d[f'mean_hidden_L{j}'][:,keep] for j in range(13)}\ndef z(X): return (X-X.mean(0))/(X.std(0)+1e-6)\ndef kcv(K, y, lams, seeds=(0,1,2)):\n    Y=np.eye(8)[y]; out={}\n    for lam in lams:\n        a=[]\n        for s in seeds:\n            f=folds(y,5,s); S=np.zeros((len(y),8))\n            for i in range(5):\n                tr=f!=i; te=f==i\n                A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0))\n                S[te]=K[np.ix_(te,tr)]@A\n            a.append((S.argmax(1)==y).mean())\n        out[lam]=round(100*np.mean(a),2)\n    return out\nZn={j:(lambda Z: Z/np.linalg.norm(Z,axis=1,keepdims=True))(z(M[j])) for j in range(13)}\nQ={j:(Zn[j]@Zn[j].T) for j in (7,8,9)}\nLk={j:z(M[j])@z(M[j]).T/M[j].shape[1] for j in range(10)}\nL=(1e-2,3e-2)\nfor top in (8,9):\n    Kq=Q[top]+1.0*Q[top]**2\n    for lins in [(2,),(2,5),(2,6),(3,6),(1,4,6),(2,4,6),(2,5,7),(1,3,5,7),(4,),(5,),(2,4),(3,5),(0,2,5)]:\n        if max(lins)>=top: continue\n        print(top,lins,kcv(Kq+sum(Lk[j] for j in lins),y,L))"
}
```

