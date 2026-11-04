## reasoning summary

I'm checking the coefficient of variation for the CLS vectors at each layer, comparing the perturbed mean-token values against the base means, and testing combinations of the layer-11 CLS and mean signals. I'm also measuring how much the mean-token values shifted relative to the baseline.



## message

Now the analysis: how class information in the new [CLS] evolves across layers, how much other tokens were disturbed (mean_k vs. base), and whether cls11' + mean11' beats the v1 combo (97.5%).

## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nD=np.load('base_feats.npz'); y=D['y']; folds=np.load('folds.npy')\nE={}\nfor L in [1,2,4,6,8,10,11]:\n    for t in ['cls','mean']:\n        E[f'{t}{L}']=np.concatenate([np.load(f'pass_6/{t}_hidden_L{L}.npy'),np.load(f'pass_7/{t}_hidden_L{L}.npy')])\nnp.savez_compressed('l0trick_feats.npz',**E)\nfor L in [1,2,4,6,8,10,11]:\n    c=E[f'cls{L}'].astype(np.float64); m=E[f'mean{L}'].astype(np.float64); mb=D[f'mean{L}'].astype(np.float64)\n    rel=np.linalg.norm(m-mb,axis=1).mean()/np.linalg.norm(mb-mb.mean(0),axis=1).mean()\n    print(L,'cls\\' lda', round(cv(c,y,folds,lda_fit,0.1)[0],4),'ridge',round(cv(c,y,folds,ridge_fit,100)[0],4),\n          '| mean\\' lda',round(cv(m,y,folds,lda_fit,0.1)[0],4),'| mean change rel',round(rel,3),flush=True)\nc11=E['cls11'].astype(np.float64); m11=E['mean11'].astype(np.float64)\nX=np.hstack([c11,m11])\nprint('cls11\\'+mean11\\' ridge',[round(cv(X,y,folds,ridge_fit,l)[0],4) for l in [100,300,1000]],'lda',[round(cv(X,y,folds,lda_fit,a)[0],4) for a in [0.1,0.3]])"
}
```

