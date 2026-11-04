## reasoning summary

I'm comparing GELU at (1,0) with lambda 0.01 against the RBF baseline, and it scores slightly higher. Smaller regularization values like 0.003 and 0.001 might do better, and (1.5,0) at lambda 0.03 also gets close. I'm also noting that the exact RBF kernel in the pooler is capped at 768 centers, while GELU in the FFN can use all of them, which makes it the stronger choice. I'm thinking about whether an approximate leave-one-out estimate could be used for tuning the GELU feature ridge.

I'm weighing a leave-one-out estimate for the GELU ridge against the RBF LOO score of 98.66. The held-out point's own center still sits among the training features, which might make the estimate slightly optimistic. I'll use the hat-matrix form in dual space and compute it for g=1, b=0 across lambda values from 0.003 to 0.03.



GELU features match or slightly beat RBF kernel ridge on the same folds (98.28 vs 98.23). I'm scanning smaller λ and computing a closed-form LOO for the GELU head to compare against RBF's 98.66.



## tool_use: run_python

```json
{
 "code": "exec(open('krr.py').read()); exec(open('lib.py').read())\nfrom scipy.special import erf\ndef gelu(a): return 0.5*a*(1+erf(a/np.sqrt(2)))\ny=np.load('base_feats.npz')['y']; X1=np.load('X1_full.npy'); F=X1[:,KEEP]\nZ,_=zs(F); dd=F.shape[1]; G=Z@Z.T/dd\ndef loo_gelu(g,b,lam):\n    H=gelu(g*G+b); Hc=H-H.mean(0); K=Hc@Hc.T\n    T=np.eye(8)[y]; T=T-T.mean(0)\n    Ainv=np.linalg.inv(K+lam*np.eye(len(y))); al=Ainv@T\n    P=T-al/np.diag(Ainv)[:,None]\n    return (P.argmax(1)==y).mean()\nfor g,b in [(1,0),(1.5,0),(0.7,0),(1,0.3)]:\n    print(g,b,[round(loo_gelu(g,b,l),4) for l in [0.001,0.003,0.01,0.03,0.1]])\n"
}
```

