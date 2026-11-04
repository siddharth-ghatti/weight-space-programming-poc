## reasoning summary

I'm weighing whether to fill the remaining six heads with content-based attention pooling, since the other candidates would just be redundant with the linear features.

I'm now looking at whether the CLS token's layer-11 state, which carries most of the signal on its own, adds anything when concatenated with the mean-pooled version, so I'll run a quick ridge check on that combination and also glance at CLS from other layers.



A 384-dim PCA of the mean loses almost nothing, which leaves six heads free for another pooled feature. I'll test whether adding [CLS]'s layer-11 state to the mean helps.



## tool_use: run_python

```json
{
 "code": "exec(open('helpers.py').read())\nlab = labels_all(); y = lab[:1024]; folds = make_folds(1024)\nlams=(0.1,0.2,0.4,0.8)\nfor l in [11]:\n    X = np.concatenate([load('mean',11), load('cls',l)],1)\n    print('mean11+cls',l, ridge_cv(X,y,lams,folds))\nX = load('mean',11)+load('cls',11)/1.0\nprint('sum', ridge_cv(X,y,lams,folds))\n"
}
```

