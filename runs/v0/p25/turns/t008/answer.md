## reasoning summary

The mean-pooled layer 11 result at 96.7% is strong, and even the bag-of-embeddings baseline is already around 96%. I'll test lower LDA alpha values (0.01, 0.03, 0.05) on layer 11 and layer 8.

I'm considering whether the routable options can capture more than mean-pooling alone, since the final-layer attention could mix CLS and mean information through a linear map of the pooled representation.

I'm thinking about combining the CLS and mean features through the residual path, with LDA on the concatenated [cls11, mean11] inputs. The idea is to add a large constant vector so LayerNorm becomes roughly linear in the pooled signal, keeping float32 precision in mind.

I'm weighing whether the extra checks are worth running: concatenated [cls11, mean11] features and combos like [mean8, mean11] or cls12+mean12, since a gain of about one point could sit inside the roughly 0.5% standard error on 1496 samples. I'll run them quickly anyway, and keep in mind that the mean-pooling choice (including or excluding CLS/SEP tokens) may matter slightly.

I'm dropping the idea of routing through the CLS residual, since changing CLS would also disturb how the other tokens attend to it. I'll move on to a quicker check: lower LDA alpha, ridge, and the concatenated feature combinations.



Mean-pooled hidden states reach about 96–97% CV accuracy at nearly every layer, versus roughly 94.5% for final-layer CLS alone. Next I'll tune the shrinkage, try ridge, and test CLS and mean-pooled combinations that could feed the final CLS.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nD=np.load('base_feats.npz'); y=D['y']; folds=np.load('folds.npy')\ndef show(name,X):\n    X=X.astype(np.float64)\n    r=[cv(X,y,folds,lda_fit,a)[0] for a in [0.01,0.03,0.1,0.2]]\n    r2=[cv(X,y,folds,ridge_fit,l)[0] for l in [10,100,1000]]\n    print(name,'LDA',' '.join(f'{v:.3f}' for v in r),'| ridge',' '.join(f'{v:.3f}' for v in r2),flush=True)\nfor k in ['mean8','mean11','mean12','cls12']: show(k,D[k])\nshow('cls11+mean11',np.hstack([D['cls11'],D['mean11']]))\nshow('mean8+mean11',np.hstack([D['mean8'],D['mean11']]))\nshow('mean0+mean11',np.hstack([D['mean0'],D['mean11']]))\nshow('cls11+mean11 sum',D['cls11']+D['mean11'])"
}
```

