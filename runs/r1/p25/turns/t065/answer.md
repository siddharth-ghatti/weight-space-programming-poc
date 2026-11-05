## reasoning summary

I'm checking how R2 changes across layers 1 to 5 at m=1536 against pass 12 at m=2048. Then I'll get the layer-5 CLS input from LN1, fit layer 5 under two targets (the standard local one and the corrected x_base minus that input), and compare their final CLS hidden error after LN2 over 1000-sample CV against what pass 12 reached at L6.



Next I'll analyze pass 13's upstream deviation at m=1536 versus m=2048, then fit layer 5 on the edited inputs under two targets and score both on held-out folds.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nexec(open('lsbuild.py').read())\ndef r2(a,b): return 1-((a-b)**2).sum()/((b-b.mean(0))**2).sum()\nfor k in range(1,6):\n    cb=np.load(f'pass_1/cls_hidden_L{k}.npy'); mb=np.load(f'pass_1/mean_hidden_L{k}.npy')\n    print(f'L{k}: m2048 cls {r2(np.load(f\"pass_12/cls_hidden_L{k}.npy\"),cb):.4f} mean {r2(np.load(f\"pass_12/mean_hidden_L{k}.npy\"),mb):.4f} | m1536 cls {r2(np.load(f\"pass_13/cls_hidden_L{k}.npy\"),cb):.4f} mean {r2(np.load(f\"pass_13/mean_hidden_L{k}.npy\"),mb):.4f}')\ndef LN(x,g,b): mu=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-mu)/np.sqrt(v+1e-5)*g+b\npre='roberta.encoder.layer.5.'\ng1=base(pre+'attention.output.LayerNorm.weight').astype(np.float64); be1=base(pre+'attention.output.LayerNorm.bias').astype(np.float64)\ng2=base(pre+'output.LayerNorm.weight').astype(np.float64); be2=base(pre+'output.LayerNorm.bias').astype(np.float64)\nz=np.load('pass_13/cls_hidden_L6.npy').astype(np.float64)\nae=LN(z,g1,be1); np.save('a5e_1536.npy',ae.astype(np.float32))\nab,W1,b1,U,b2=layer_data(5,base)\nGb=gelu(ab@W1.T+b1); xb=ab+Gb@U.T+b2; hb=LN(xb,g2,be2)\nprint('a5 rel err', np.linalg.norm(ae-ab)/np.linalg.norm(ab-ab.mean(0)))\nGe=gelu(ae@W1.T+b1)\nfolds=np.load('folds.npy'); nf=folds.max()+1\ndef fit_eval(m, lam, target):\n    he=np.zeros_like(hb)\n    for f in range(nf):\n        tr=folds!=f; te=~tr\n        mu=Ge[tr].mean(0); Gc=Ge[tr]-mu\n        sc=(U**2).sum(0)*(Gc**2).sum(0)\n        keep=np.sort(np.argsort(-sc)[:m])\n        T=target[tr]; tm=T.mean(0)\n        X=Gc[:,keep]\n        # ridge toward original U cols: solve for D=Un-U[:,keep]\n        R=T-tm-X@U[:,keep].T\n        D=np.linalg.solve(X.T@X+lam*np.eye(m),X.T@R).T\n        Un=U[:,keep]+D\n        Fe=(Ge[te][:,keep]-mu[keep])@Un.T+tm\n        he[te]=LN(ae[te]+Fe,g2,be2)\n    return r2(he,hb)\nFloc=Ge@U.T+b2; Tcor=xb-ae\nfor m in (1536,):\n    for lam in (3,10,30,100):\n        print(m,lam,'local',round(fit_eval(m,lam,Floc),4),'corrected',round(fit_eval(m,lam,Tcor),4))\nprint('no-prune full FFN on ae: h6 R2', round(r2(LN(ae+Floc,g2,be2),hb),4), ' pass12 L6 cls R2', round(r2(np.load('pass_12/cls_hidden_L6.npy'),np.load('pass_1/cls_hidden_L6.npy')),4))\n"
}
```

