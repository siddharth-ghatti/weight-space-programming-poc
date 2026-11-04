## reasoning summary

I'm computing the regression compensations and the head-0 rank-56 approximation next. That means checking the approximation error for head 0's content map, weighted by the M0 covariance, and the R² values for the residual regressions from their respective keep sets.

I'm working out how the base model's layer-0 input relates to the normalized exposure Z, which is Z times the base gamma plus beta.



Next I'll test a rank-56 approximation of head 0's uniform-attention content map against the actual M0 weighting, and check how well layers 0/1's reserved-dim inputs can be regressed from the other dims.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0, str(WORK))\nfrom lib import *\nd=np.load('base_feats.npz'); M0=d['mean_hidden_L0'].astype(np.float64)\nP='bert.encoder.layer.0.attention.'\nWv=W(P+'self.value.weight').astype(np.float64); Wo=W(P+'output.dense.weight').astype(np.float64)\nM=Wo[KEEP][:,0:64]@Wv[0:64]   # 760x768\nmu=M0.mean(0); Sig=np.cov(M0.T)+1e-6*np.eye(768)\nev,EV=np.linalg.eigh(Sig); Sh=EV@np.diag(np.sqrt(np.maximum(ev,0)))@EV.T; Shi=EV@np.diag(1/np.sqrt(np.maximum(ev,1e-6)))@EV.T\nU,S,Vt=np.linalg.svd(M@Sh); print('weighted sv (top/56/64):',S[0].round(3),S[55].round(4),S[63].round(5), 'energy kept by 56: %.5f'%((S[:56]**2).sum()/(S**2).sum()))\n# relative to the head's own output variation and residual stream\nout=(M0-mu)@M.T; print('head out var',(out**2).sum(1).mean().round(4))\n# regressions\ndef regR2(Xk, Xr, lam=1.0):\n    mk=Xk.mean(0); mr=Xr.mean(0); A=Xk-mk; B=Xr-mr\n    Bc=np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]), A.T@B); pred=A@Bc\n    return 1-((B-pred)**2).sum(0)/(B**2).sum(0), Bc, mr-mk@Bc\nmsk10=np.load('pass_10/attention_mask.npy').astype(bool); Z=np.load('pass_10/token_hidden_L1.npy')[msk10].astype(np.float64)\nX0=Z*base(P+'output.LayerNorm.weight')+base(P+'output.LayerNorm.bias')\nr2,_,_=regR2(X0[:,KEEP],X0[:,RES]); print('X0 res R2',r2.round(3))\nmsk23=np.load('pass_23/attention_mask.npy').astype(bool); H1=np.load('pass_23/token_hidden_L1.npy')[msk23].astype(np.float64)\nr2,_,_=regR2(H1[:,KEEP],H1[:,RES]); print('h1 res R2',r2.round(3))\nmsk11=np.load('pass_11/attention_mask.npy').astype(bool); Z1=np.load('pass_11/token_hidden_L2.npy')[msk11].astype(np.float64)\nX1=Z1*base('bert.encoder.layer.1.attention.output.LayerNorm.weight')+base('bert.encoder.layer.1.attention.output.LayerNorm.bias')\nr2,_,_=regR2(X1[:,KEEP],X1[:,RES]); print('X1 res R2',r2.round(3))\n# importance of RES reads: fraction of pre-activation variance from RES columns\nw1=base('bert.encoder.layer.0.intermediate.dense.weight').astype(np.float64)\npre_res=(X0[:,RES]-X0[:,RES].mean(0))@w1[:,RES].T; pre=(X0-X0.mean(0))@w1.T\nprint('L0 FFN preact var frac from RES: %.4f'%((pre_res**2).sum()/(pre**2).sum()))"
}
```

