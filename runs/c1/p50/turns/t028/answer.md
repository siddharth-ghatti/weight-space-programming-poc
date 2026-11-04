## reasoning summary

I'm checking whether the gap to the RBF's 97.4% comes from using only 768 anchors or from the normalization choice, and I'm planning an exact KRR test with the exp(a·z·z') kernel on all points to separate the two effects.

I'm now testing a tanh(2s−1) kernel on all points without anchors, and comparing it against the exp kernel with Nyström, to see whether the shape of the kernel is what matters.



## message

Standardizing the features didn't close the gap. Next I'll separate the two possible causes: the kernel shape (tanh vs exp) or using only 768 anchors (Nyström) instead of all training points.

## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\nd=load_base(); y=labels()\nM=lambda l: d[f'mean_hidden_L{l}'][:,KEEP]\nX=M(0); Z=X-X.mean(0); Z=Z/np.sqrt((Z**2).sum(1).mean()); G=Z@Z.T; sq=np.diag(G)\nprint('rms-norm stats: diag', sq.min(), sq.max(), 'offdiag pct', np.percentile(G[np.triu_indices(len(G),1)],[1,50,99]))\nD2=sq[:,None]+sq[None,:]-2*G; med=np.median(D2); print('med',med)\nprint('rbf', krr_cv(np.exp(-D2/med),y,(0.003,0.01),k=10,seeds=(0,)))\nprint('exp(G/med*2)', krr_cv(np.exp(2*G/med),y,(0.003,0.01,0.03),k=10,seeds=(0,)))\nfor a,b in [(2,-1),(1,0),(1,-1),(0.7,-0.5)]:\n    print('tanh full',a,b, krr_cv(np.tanh(a*G+b)+1,y,(0.003,0.01,0.03,0.1),k=10,seeds=(0,)))\nZc=Z/np.linalg.norm(Z,axis=1,keepdims=True); Gc=Zc@Zc.T\nprint('cos-exp', krr_cv(np.exp(Gc),y,(0.003,0.01,0.03),k=10,seeds=(0,)))\nprint('lin',krr_cv(G,y,(0.1,0.3,1),k=10,seeds=(0,)))"
}
```

