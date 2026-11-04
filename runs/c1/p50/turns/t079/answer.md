## reasoning summary

I'm comparing the pruned results at m=1024: keeping only L9 gives 97.07 against a base of 97.61, and adding M0 brings it to 98.10 versus 98.44, so the drop is roughly 0.4 points on a sample that's fairly noisy. I'm thinking the full 1496-sample set with more training data should narrow that gap, and I'm weighing a few ways to improve it, starting with a larger m of 1536.

I'm weighing ways to recover the gap from pruning, and one idea is to refit each layer's FFN on the pruned-upstream inputs so it targets the base outputs, though that costs extra passes. Another is to combine features from intermediate layers, since the final pooled layer only sees M9. I'm also considering carrying M0 information forward through a few reserved dimensions, which would need cross-fitting to avoid leakage.

I'm checking whether M0 can be linearly recovered from the layer-8 token states, since M0+M9 outperforming M9 alone hints that lexical information is getting lost. I'm considering a ridge map R from M9 to M0, fit across sentences without labels so it doesn't leak, and then testing a kernel on the concatenation of M9 and R·M9. It might help by reweighting directions in the metric, or it might not.

I'm considering whether the reserved dimensions could be expanded to carry more of M0, though that would change the layer's function. A uniform-attention head in layer 0 could also pass the mean embedding forward at the CLS position, but later layers would alter it and other positions would receive it too, so I'm turning to a cleaner option: reading the layer-9 residual stream at the CLS token directly.

I'm considering whether the CLS token's layer-9 features, or a kernel over M0, M9 and the CLS state, could recover some of the gap. I'm also thinking the pool layer's FFN at CLS, with its m units, might work as a kernel machine of its own, alongside the pooler's tanh.

I'm weighing how to fit M0-like information into the pool layer's 768 value dimensions, since all of them are currently spent on M9. Splitting them, say 384 for a PCA of M9 and the rest for a projection that approximates the embeddings, looks like the main tradeoff to test.

I'm planning to test a kernel on the 1496-sample set using M9 with a PCA reduction, comparing it against a version that adds a ridge-mapped projection of M9 toward M0, and checking whether that beats the M0+M9 baseline.

If M0 is close to R·M9 (high R²), then the pair [R·M9, M9] should carry roughly the same information as [M0, M9].



Dropping m=1024 costs about 0.3–0.5 points, and M0 can't be read directly at the final layer, so I'll test whether it's linearly recoverable from M9 on base features.



## tool_use: run_python

```json
{
 "code": "exec(open('lib.py').read())\ny=labels(); B=load_base()\ndef cen(X): X=X[:,KEEP].astype(np.float64); return X-X.mean(0)\ndef bn(X): return X/np.sqrt((X**2).sum(1).mean())\ndef kcv(Z, gs=(0.5,1), lams=(0.003,0.01,0.03), k=10):\n    Z=Z/np.linalg.norm(Z,axis=1,keepdims=True); S=Z@Z.T\n    return max(max(krr_cv(np.exp(g*(S-1)),y,lams=lams,k=k,seeds=(0,)).values()) for g in gs)\nM0=cen(B['mean_hidden_L0']); M9=cen(B['mean_hidden_L9']); C9=cen(B['cls_hidden_L9'])\n# R: M9 -> M0 ridge (unsupervised)\nfor lam in [1,10,100]:\n    R=np.linalg.solve(M9.T@M9+lam*np.eye(M9.shape[1]), M9.T@M0)\n    # CV R^2 quickly (2-fold)\n    h=np.arange(1496)%2==0\n    R2=np.linalg.solve(M9[h].T@M9[h]+lam*np.eye(760), M9[h].T@M0[h]); pr=M9[~h]@R2\n    print('lam',lam,'heldout R2 %.3f'%(1-((pr-M0[~h])**2).sum()/((M0[~h]-M0[~h].mean(0))**2).sum()))\nR=np.linalg.solve(M9.T@M9+10*np.eye(760), M9.T@M0)\nprint('M9', kcv(bn(M9)))\nprint('M0+M9', kcv(np.concatenate([bn(M0),bn(M9)],1)))\nprint('RM9+M9', kcv(np.concatenate([bn(M9@R),bn(M9)],1)))\nprint('M9+C9', kcv(np.concatenate([bn(M9),bn(C9)],1)))\nprint('M0+M9+C9', kcv(np.concatenate([bn(M0),bn(M9),bn(C9)],1)))\n"
}
```

