## reasoning summary

I'm going to check the quality on these 1024 items using the same metric from my notes, comparing the base and m1024/m1536 variants on both the kernel ridge and joint setups.

I'm now checking the kernel ridge and joint setups on the same 1024 items, using the RBF and centered-linear kernels, and I'll compare against the base and m1024/m1536 scores from my notes.



Next I'll benchmark the new sequential-m1024 features on the same 1,024 ids, using the earlier 5-seed 10-fold KRR metric, and I'll also test the 8-dim carried RES features as the linear block.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom lib import labels, krr_cv, KEEP, RES\ny=labels()[:1024]\nM9=np.load('pass_34/mean_hidden_L9.npy').astype(np.float64); C9=np.load('pass_34/cls_hidden_L9.npy').astype(np.float64)\nM0=np.load('pass_34/mean_hidden_L0.npy').astype(np.float64)\ndef cosK(X,g):\n    Xc=X-X.mean(0); Xn=Xc/np.linalg.norm(Xc,axis=1,keepdims=True); S=Xn@Xn.T; return np.exp(g*(S-1))\ndef lin(X):\n    Xc=X-X.mean(0); return Xc/np.sqrt((Xc**2).sum(1).mean())\nK9=cosK(M9[:,KEEP],1.0)\nprint('K9', krr_cv(K9,y,[0.003,0.01,0.03],k=10,seeds=range(5)))\nL0=lin(M0); Kj=K9+0.5*L0@L0.T\nprint('K9+linM0', krr_cv(Kj,y,[0.003,0.01,0.03],k=10,seeds=range(5)))\nfor nm,R in [('meanRES',M9[:,RES]),('clsRES',C9[:,RES])]:\n    Lr=lin(R)\n    for c in [0.5,2,8]:\n        print(nm,c, krr_cv(K9+c*Lr@Lr.T,y,[0.01,0.03],k=10,seeds=range(3)))\n# carry quality: corr of RES with designed scores\nd=np.load('carry_prov.npz'); ec=(M0-d['mu0'])@d['wraw']\nprint('corr meanRES', [round(np.corrcoef(M9[:,RES][:,j],ec[:,j])[0,1],3) for j in range(8)])\nprint('corr clsRES', [round(np.corrcoef(C9[:,RES][:,j],ec[:,j])[0,1],3) for j in range(8)])\n"
}
```

