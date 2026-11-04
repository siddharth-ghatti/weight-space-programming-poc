## reasoning summary

I'm comparing the cls_hidden_L12 activations against my LayerNorm estimate, and after that I'll fit the classifier on the pooled data.



## message

Now I'll compare the actual final CLS to my prediction, then fit the head on the real pooled outputs (cross-validated).

## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\ny=np.load('base_feats.npz')['y']; folds=np.load('folds.npy')\nh12=np.concatenate([np.load('pass_3/cls_hidden_L12.npy'),np.load('pass_4/cls_hidden_L12.npy')]).astype(np.float64)\npooled=np.concatenate([np.load('pass_3/pooled.npy'),np.load('pass_4/pooled.npy')]).astype(np.float64)\nLn=np.load('LN_est_v1.npy')\nerr=np.abs(h12-Ln); print('max abs diff vs predicted', err.max(), 'mean', err.mean())\nprint('pooled range', pooled.min(), pooled.max(), np.abs(pooled).mean())\nnp.save('v1_pooled.npy',pooled.astype(np.float32))\nfor a in [0.05,0.1,0.2,0.4]:\n    acc,pred=cv(pooled,y,folds,lda_fit,a); print('lda',a,round(acc,4),round(macro_f1(y,pred),4))\nfor l in [10,30,100,300]:\n    acc,pred=cv(pooled,y,folds,ridge_fit,l); print('ridge',l,round(acc,4),round(macro_f1(y,pred),4))"
}
```

