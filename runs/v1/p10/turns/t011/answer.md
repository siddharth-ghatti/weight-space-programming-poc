## reasoning summary

I'm seeing PCA look worse at 512 dims, and I suspect the per-column standardization in ridge_fit is inflating low-variance components into noise, so the comparison isn't fair. I want to rerun ridge on the raw features without per-dimension standardization, since outlier dimensions in the LayerNorm'd features might still make it matter.

I'm weighing whether the standardization step even matters here, since the pooler can apply any affine map anyway. The projection onto the top-k PCA dimensions is linear, so what really counts is whether that subspace keeps the information. I'll redo it: standardize first, then PCA, then fit ridge without another standardization pass.



The PCA comparison was unfair because my ridge standardizes each column and inflates the tiny-variance components. I'm redoing it by standardizing first, projecting, then running ridge without re-standardizing.



## tool_use: run_python

```json
{
 "code": "exec(open('helpers.py').read())\nlab = labels_all(); y = lab[:1024]; folds = make_folds(1024)\nm = load('mean',11)\ndef ridge_cv_raw(X, y, lams, folds):\n    res={}\n    for lam in lams:\n        c=0\n        for f in folds:\n            tr=np.setdiff1d(np.arange(len(y)),f); mu=X[tr].mean(0)\n            Xt=X[tr]-mu; Y=np.eye(8)[y[tr]]\n            Wt=np.linalg.solve(Xt.T@Xt+lam*len(tr)*np.eye(X.shape[1]), Xt.T@(Y-Y.mean(0)))\n            c+=(((X[f]-mu)@Wt).argmax(1)==y[f]).sum()\n        res[lam]=round(c/len(y),4)\n    return res\nms = (m-m.mean(0))/m.std(0)\nU,S,Vt = np.linalg.svd(ms, full_matrices=False)\nlams=(0.05,0.1,0.2,0.4)\nprint('full', ridge_cv_raw(ms,y,lams,folds))\nfor k in [128,256,384,512]:\n    print(k, ridge_cv_raw(ms@Vt[:k].T,y,lams,folds))\n"
}
```

