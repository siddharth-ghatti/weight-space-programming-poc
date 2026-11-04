## reasoning summary

Plain linear stacking isn't enough, so the M0 information has to come in jointly and nonlinearly. I'm checking whether a simple additive kernel, K0 + K9 averaged, gets there, since that would mean a separate kernel machine on M0 with some nonlinearity somewhere.



Stacked linear scores add nothing, so the M0 gain must come from a joint nonlinear combination; I'm testing whether an additive kernel (K0 + K9) matches the joint product kernel.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0, str(WORK))\nfrom lib import *\ny=labels(); d=np.load('base_feats.npz')\ndef cn(X):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.linalg.norm(X,axis=1,keepdims=True)\nZ0=cn(d['mean_hidden_L0']); Z9=cn(d['mean_hidden_L9']); Z1=cn(d['mean_hidden_L1'])\nK0=np.exp(Z0@Z0.T-1); K9=np.exp(Z9@Z9.T-1); K1=np.exp(Z1@Z1.T-1)\nprint('K9',krr_cv(K9,y,[0.005],k=10,seeds=(0,1,2)))\nprint('K0*K9 (joint)',krr_cv(np.exp((Z0@Z0.T+Z9@Z9.T)/2-1),y,[0.005],k=10,seeds=(0,1,2)))\nfor w in [0.3,0.5]:\n    print('additive w',w,krr_cv((1-w)*K9+w*K0,y,[0.003,0.005,0.01],k=10,seeds=(0,1,2)))\nprint('additive K1',krr_cv(0.5*K9+0.5*K1,y,[0.005],k=10,seeds=(0,1,2)))\n# linear-in-M0 kernel added\nprint('K9 + lin0',krr_cv(K9+0.5*(Z0@Z0.T),y,[0.005],k=10,seeds=(0,1,2)))"
}
```

