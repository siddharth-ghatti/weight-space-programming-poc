## reasoning summary

I'm noting that the sharper features do much worse under plain ridge, probably because of the penalty or scale. The ridge penalty here applies to the 768-dim L0 features in CV, not the 8-dim linear carry in the real model. Pivchol only took 0.84 s, which I want to double-check.

I'm settling on a=1, b=0 with plain ridge at lam 0.01–0.1 and c0=0.5, and I want to check lam=0.03 across several seeds and a smaller anchor count to see how sensitive the pool is. I'm also going to run the leaky real-pipeline version with z_RES as the linear block, just as a training-fit sanity check.

Setting the attention output so the CLS token carries the centered M9 features, with the residual added back. Then the FFN's first layer: each anchor unit should be a scaled direction, but LayerNorm pins the norm of the input vector, so I have to work out what the cosine actually becomes in the real model.

Working out the feature computation for the real model: since LayerNorm fixes the full 768-dim norm, the KEEP part's norm is nearly constant, so I can set each anchor unit as the normalized KEEP direction scaled by 1/sqrt(768). Then the pre-activation is just the dot product with the LayerNorm output, which I'll compute exactly as the model would rather than using the normalized version.

Now I'm setting the residual-linear pairs: one unit gets the positive direction on each RES dimension and its twin gets the negative, so their GELU difference recovers the linear term. Then the output weights for those RES rows will map the anchor and pair activations back to the class scores, with the residual connection accounted for.

I'm working out how the output projection has to map the class scores back onto the RES dimensions, since the residual connection adds z_RES to those same dimensions and that has to be subtracted out.

Now I'm checking the output pooling layer: the dense rows for the eight class dimensions take a centered mapping over the residual-linear dimensions, with everything else zeroed. The classifier just passes those eight pooled values through as the class scores, with no bias.

Still setting the scale for the pooler output: I'll start with s_p = 1 and check that the pre-tanh values stay within about ±3 in numpy before finalizing.

Setting the ridge penalty for the final fit: the 1008 pool features get one lambda, while the 8-dim z_RES block needs its own scaling, since its magnitude is much smaller than the L0 features.

I'm working out the penalty for the 8-dim linear block: rescaling z_RES to unit mean squared norm so its penalty matches the L0 features, then converting the coefficients back. Before settling on a final lambda, I want to check multi-seed CV across a range of values.

I'm checking the ridge penalty across a few values and seeds, with 1008 anchors and then 752 for the m=768 option. Once that looks right I'll build the model and confirm the fit reaches roughly 100% training accuracy with the linear block included.



Plain GELU features with primal ridge reach 98.4% on seed 0, close to the exact-kernel level and clearly ahead of the sharper shapes. I'm checking the λ choice across more seeds and how sensitive the result is to fewer anchors.



## tool_use: run_python

```json
{
 "code": "import sys, time; sys.path.insert(0,'.')\nfrom poolfit import *\nfrom lib import labels, folds\ny=labels(); Y=np.eye(8)[y]\nF=np.load('feats_seq1024.npz'); M9=F['mean_hidden_L9'].astype(np.float64); C9=F['cls_hidden_L9'].astype(np.float64); M0=F['mean_hidden_L0'].astype(np.float64)\nmu9=M9.mean(0); Z=pool_z(M9,C9,20.0,mu9,1.0)\nL0=M0-M0.mean(0); L0/=np.sqrt((L0**2).sum(1).mean())\nZk=Z[:,KEEP]; Zn=Zk/np.linalg.norm(Zk,axis=1,keepdims=True); S=Zn@Zn.T\nKex=np.exp(S-1)\ndef ridge(X, Y, pen):\n    mu=X.mean(0); Xc=X-mu; Ym=Y.mean(0)\n    coef=np.linalg.solve(Xc.T@Xc+np.diag(pen), Xc.T@(Y-Ym)); return coef, Ym-mu@coef\nres={}\nfor seed in [1,2,3]:\n    f=folds(y,10,seed); pivs=[]\n    for i in range(10):\n        tr=np.where(f!=i)[0]; pivs.append(tr[pivchol(Kex[np.ix_(tr,tr)],1008)])\n    for na in [1008,752]:\n        for lam in [0.01,0.03,0.1]:\n            for c0 in [0.25,0.5,1.0]:\n                Sc=np.zeros((1496,8))\n                for i in range(10):\n                    tr=f!=i; te=f==i; P=gelu(S[:,pivs[i][:na]])\n                    X=np.concatenate([P,L0],1); pen=np.r_[lam*np.ones(na), lam/c0*np.ones(768)]\n                    coef,b0=ridge(X[tr],Y[tr],pen); Sc[te]=X[te]@coef+b0\n                res.setdefault((na,lam,c0),[]).append((Sc.argmax(1)==y).mean())\nfor k,v in sorted(res.items()): print(k, round(float(np.mean(v)),4))\n"
}
```

