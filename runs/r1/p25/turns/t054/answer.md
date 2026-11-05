## reasoning summary

I'm comparing how the error drops for L0 as the training set grows from 8 to 16 to 28 ids, and both the selection and LS variants improve. Extrapolating, 64 and 128 ids should push it down further, but with diminishing returns. I'm guessing the floor at m=2048 sits around 0.025 for L0, though I'm not sure yet.

So at m=2048 the error should still be around 3-6% per hard layer, which raises the question of whether that's acceptable. I'd rather test the LS-refit version at m=2048 against the original head and see if agreement holds at 0.98 or above, falling back to m=2304 or 2560 if not. Layer 5 is different since only the CLS token matters, so I'd fit least squares on the 1000 CLS samples against the FFN output with cross-validated regularization.

I'm checking the EOS error at m=2048 for layer 1 and considering a higher weight for EOS in the least-squares fit, since it may act as an attention sink like CLS. I also want to verify that the pass 11 comparison is honest, given that its edit was built from Gaussian statistics rather than data.

Now I'm weighing a plan to build the m=2048 least-squares edit for layers 0-4 from all 32 ids and fit layer 5 on the 1000 CLS samples with cross-validated regularization, then run a pass to check it.

I'm writing a builder function that takes a layer index, width, regularization, and weights, fitting least squares on the base FFN inputs for layers 0-4 and on the CLS samples for layer 5. Before saving I'll clear out any existing FFN edits, and for layer 5 I'll pick the regularization strength by cross-validation.



More data lowers the layer 0 error predictably, so before spending more passes I'll calibrate with one real pass, then fit layer 5's head on the 1000 <s> FFN inputs from pass 10 and pick λ by held-out folds.



## tool_use: run_python

```json
{
 "code": "import numpy as np, scipy.special as sps\ndef gelu(z): return 0.5*z*(1+sps.erf(z/np.sqrt(2)))\na=np.load('true_a5_cls.npy').astype(np.float64)\nl=5; pre=f'roberta.encoder.layer.{l}.'\nW1=base(pre+'intermediate.dense.weight').astype(np.float64); b1=base(pre+'intermediate.dense.bias').astype(np.float64)\nU=base(pre+'output.dense.weight').astype(np.float64); b2=base(pre+'output.dense.bias').astype(np.float64)\nG=gelu(a@W1.T+b1); F=G@U.T+b2\n# sanity: LN2(a+F) should equal base cls_hidden_L6\ng2=base(pre+'output.LayerNorm.weight'); be2=base(pre+'output.LayerNorm.bias')\ndef LN(x,g,b): mu=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-mu)/np.sqrt(v+1e-5)*g+b\nh6=LN(a+F,g2,be2); hb=np.load('pass_1/cls_hidden_L6.npy')\nprint('recon check', np.abs(h6-hb).max(), np.abs(hb).max())\nfolds=np.load('folds.npy'); nf=folds.max()+1\nfor m in (1536,2048):\n  for lam in (1,10,100,1000):\n    num=0;den=0\n    for f in range(nf):\n        tr=folds!=f; te=~tr\n        mu=G[tr].mean(0); Gc=G[tr]-mu\n        sc=(U**2).sum(0)*(Gc**2).sum(0)\n        keep=np.sort(np.argsort(-sc)[:m]); pr=np.setdiff1d(np.arange(3072),keep)\n        A=Gc[:,keep].T@Gc[:,keep]; B=Gc[:,keep].T@(Gc[:,pr]@U[:,pr].T)\n        D=np.linalg.solve(A+lam*np.eye(m),B).T\n        Fe=F[te]-((G[te][:,pr]-mu[pr])@U[:,pr].T)+(G[te][:,keep]-mu[keep])@D.T\n        num+=((Fe-F[te])**2).sum(); den+=((F[te]-F[tr].mean(0))**2).sum()\n    print(m,lam,round(num/den,5))\n"
}
```

