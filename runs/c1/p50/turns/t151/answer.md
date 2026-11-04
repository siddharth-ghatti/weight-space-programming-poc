## reasoning summary

I'm noticing the 99.8% score is inflated by label leakage: the carry weights were fit on all 1496 labels, including the test folds. To get an honest estimate, I need to refit the linear carry on each training fold only, since it's approximately a linear function of M0.

Now I'm checking whether the joint linear-plus-K9 proxy is a fair stand-in for the final model, since the real pipeline fixes an 8-dimensional projection and a pool from all 1496 labels. Both parts would have to be refit inside each fold for an honest number, and the proxy doesn't do that.

I'm weighing whether the honest test estimate for the pruned kernel-plus-linear joint proxy, around 98.0 to 98.3 against a 97.8 threshold, leaves enough margin. I'm also considering whether adding CLS features or changing the kernel could squeeze out more accuracy.

I'm considering whether a kernel on the concatenated M9 and CLS features, scaled by a weight, could add accuracy, and I want to try a few weight and gamma values quickly. I'm also wondering whether M0 could enter the kernel directly, but the pool layer only sees M9, the CLS features, and the 8 carry dimensions, so that would need the carry to be wider.

I'm going to check whether a kernel on the CLS features, weighted against M9, adds accuracy, and try a few values of g and the CLS weight on the 1024 subset, since that's cheap to run. Then I'll decide whether it's worth pursuing before the second read.



The 99.8% is leakage, since the carried scores were fitted on the CV test folds; the honest figure is 97.97 from the joint fit. I'm checking whether a different kernel width or adding the CLS state helps before settling the pool design.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom lib import labels, krr_cv, KEEP, RES\ny=labels()[:1024]\nM9=np.load('pass_34/mean_hidden_L9.npy').astype(np.float64); C9=np.load('pass_34/cls_hidden_L9.npy').astype(np.float64)\nM0=np.load('pass_34/mean_hidden_L0.npy').astype(np.float64)\ndef cosK(X,g):\n    Xc=X-X.mean(0); Xn=Xc/np.linalg.norm(Xc,axis=1,keepdims=True); S=Xn@Xn.T; return np.exp(g*(S-1))\ndef lin(X):\n    Xc=X-X.mean(0); return Xc/np.sqrt((Xc**2).sum(1).mean())\nL0=lin(M0); LL=0.5*L0@L0.T\nfor g in [0.5,2.0]:\n    print('g',g, krr_cv(cosK(M9[:,KEEP],g)+LL,y,[0.003,0.01,0.03],k=10,seeds=range(3)))\nA=lin(M9[:,KEEP]); B=lin(C9[:,KEEP])\nfor w in [0.1,0.2,0.35]:\n    X=np.concatenate([A,w*B],1)\n    print('cls w',w, krr_cv(cosK(X,1.0)+LL,y,[0.003,0.01],k=10,seeds=range(3)))\n# raw (uncentered-around-mean x) variant: x = c*(M9-mu)+C9  (what pool attention gives), c large/small\nfor c in [3.0,6.0]:\n    X=c*(M9[:,KEEP]-M9[:,KEEP].mean(0))+C9[:,KEEP]\n    Xn=X-X.mean(1,keepdims=True); Xn/=np.linalg.norm(Xn,axis=1,keepdims=True)\n    K=np.exp(1.0*(Xn@Xn.T-1))\n    print('pool-x c',c, np.linalg.norm(c*(M9[:,KEEP]-M9[:,KEEP].mean(0)),axis=1).mean()/np.linalg.norm(C9[:,KEEP],axis=1).mean(), krr_cv(K+LL,y,[0.003,0.01],k=10,seeds=range(3)))\n"
}
```

