## reasoning summary

The exact RBF on this z plus linear term gets 98.46, while the exact GELU(cos) kernel gets 98.33, so GELU(cos) gives up about 0.13 against exp. I'm wondering whether I can get closer to exp(cos-1) by using GELU(a t + b) with a scale and bias, since that can match the convex, increasing shape of exp over t in [-1, 1].

I'm weighing whether a scaled and shifted GELU, GELU(a·t + b), can track the convex, increasing shape of exp(t−1) better than plain GELU(cos). My plan is to scan a few values of a and b with exact kernel ridge regression over all anchors, and I'm noting that a large a would make the kernel behave like a truncated linear (ReLU-style) one, which may not be PSD.

I'm checking a more aggressive setting now, with a = 3 and b = -2, which puts the kernel's transition near the top of the range. The values at t = 1, 0, and -1 come out roughly 0.84, -0.045, and 0, so the shape looks like a soft threshold that could resemble exp(3(t-1)).



Exact KRR on the realisable z reaches 98.46%, but the GELU(cos) kernel gets only 98.33%, a gap of about 0.13. I'm scanning the (a, b) parameters of GELU(a·cos + b) with exact KRR to find a shape closer to exp(g(cos−1)).



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom poolfit import *\nfrom lib import labels, krr_cv\ny=labels()\nF=np.load('feats_seq1024.npz'); M9=F['mean_hidden_L9'].astype(np.float64); C9=F['cls_hidden_L9'].astype(np.float64); M0=F['mean_hidden_L0'].astype(np.float64)\nmu9=M9.mean(0); Z=pool_z(M9,C9,20.0,mu9,1.0)\nL0=M0-M0.mean(0); L0/=np.sqrt((L0**2).sum(1).mean()); LL=L0@L0.T\nZk=Z[:,KEEP]; Zn=Zk/np.linalg.norm(Zk,axis=1,keepdims=True); S=Zn@Zn.T\nprint('S offdiag pct', np.percentile(S[np.triu_indices(1496,1)],[1,50,99]))\nres={}\nfor a,b in [(2,0),(2,-1),(3,-1.5),(3,-2),(4,-3),(6,-5),(2,1)]:\n    K=gelu(a*S+b); K=K/np.mean(np.diag(K))\n    r=krr_cv(K+0.5*LL,y,[0.003,0.01,0.03],k=10,seeds=range(2)); res[(a,b)]=r; print(a,b,r, 'mineig', np.linalg.eigvalsh(K)[:1])\n"
}
```

