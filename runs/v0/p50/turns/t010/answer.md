## reasoning summary

The 97.2% from mean pooling at layer 11 beats the 94.1% from CLS at layer 12, and mean pooling of hidden_states[11] is what the final layer's attention could compute anyway. I'm sketching a layer 11 setup where zeroing the query weight and bias makes attention uniform over non-masked tokens, and I still need to check whether the key bias changes that, since with q = 0 every score would be zero regardless.

Setting the output projection to a scaled identity makes the layer's attention output roughly the mean of layer 11, so the residual CLS contribution becomes small. I'm also checking whether the mean includes the CLS and SEP tokens, since uniform attention would count them and the harness might not.

I'm finishing the layer 11 plan: after editing, I'll read the layer 12 CLS outputs again, fit ridge on those, and set the pooler and classifier before predicting to check the result.

Next I'm checking the CV score for the normalized features, since layer-norm may shift the probe's accuracy a little from the raw mean.

I'm checking whether combining mean pooling across several layers, say 8 through 12, improves the CV score over the single layer-11 mean. Realizing that would take more than one pooling point, which makes it harder to build into the model. I'll also try LDA with shrinkage as a closed-form alternative to ridge and sweep the regularization strength more finely.



Mean pooling of hidden_states[11] scores 97.2% CV, against 94.1% for the final CLS. Next I'm testing LayerNorm on that mean, layer combinations, a finer lambda grid, and shrinkage LDA.



## tool_use: run_python

```json
{
 "code": "lab=np.load('feats/y.npy'); folds=np.load('feats/folds.npy')\ndef cv(X,y,lam,std=True):\n    pred=np.zeros(len(y),int)\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        mu=X[tr].mean(0); sd=(X[tr].std(0)+1e-6) if std else np.ones(X.shape[1])\n        Xt=(X[tr]-mu)/sd; Xe=(X[te]-mu)/sd\n        Y=np.eye(8)[y[tr]]-1/8\n        Wt=np.linalg.solve(Xt.T@Xt+lam*np.eye(X.shape[1]), Xt.T@Y)\n        pred[te]=(Xe@Wt).argmax(1)\n    return (pred==y).mean()\ndef lnorm(X): X=X-X.mean(1,keepdims=True); return X/np.sqrt((X**2).mean(1,keepdims=True)+1e-12)\nM={l:np.load(f'feats/mean_L{l}.npy') for l in range(13)}\nfor lam in [30,60,100,200,400]:\n    print(lam, 'm11',round(cv(M[11],lab,lam),4),'norm m11',round(cv(lnorm(M[11]),lab,lam),4),'m11 nostd',round(cv(M[11],lab,lam/10,std=False),4))\nX=np.concatenate([M[l] for l in [6,8,10,11,12]],1)\nfor lam in [100,300,1000,3000]: print('concat',lam,round(cv(X,lab,lam),4))\nX=np.concatenate([M[0],M[11]],1)\nfor lam in [100,300,1000]: print('m0+m11',lam,round(cv(X,lab,lam),4))\n"
}
```

