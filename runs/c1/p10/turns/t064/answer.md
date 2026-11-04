## reasoning summary

I'm comparing the 20-fold scores: RBF comes out at 98.70, a bit ahead of the GELU variants at 98.53 and 98.63. I'm thinking about trying a smaller lambda and a different g for GELU, and I'll also watch for float32 precision problems from large coefficients. I'm also wondering whether the exact RBF can be realized through exp.

I'm checking whether the pooler's tanh can stand in for exp, but it only has 768 units against the 1496 centers I'd need, so a two-stage version won't reach the exact RBF. Instead I'm looking at whether GELU can mimic exp near zero: for small g its expansion is quadratic, and matching exp(0.8u) suggests g around 0.5, possibly with two units per center to tighten the fit.

I'm now noting that a two-unit-per-center setup, with GELU at +g and -g, gives separate linear and even parts. The catch is that free per-unit coefficients make it a different, more overfit-prone model, but tying the output weights to a fixed combination c_r would let the FFN realize a chosen kernel built from several GELU units.

I'm working out how to fold the per-center and per-sample scale factors from the RBF expansion into the output coefficients, since the per-sample factor doesn't affect the argmax. That leaves the FFN output as a weighted sum over units, which I'm about to write out.

I'm checking how closely the GELU combination has to match exp(2γu) over the likely range of u, roughly -0.6 to 1.6. I expect a two-term fit to get within about 1e-3 relative error, and I'm wondering whether even one term with a scale and offset would be enough.

I'm checking whether a constant offset in the kernel can be absorbed into the output bias, which it can since it only adds a fixed amount per class. The linear-in-u term is harder, because it isn't a bias and the units don't obviously provide it, so I'm going to fit the kernel as a constant plus a weighted sum of GELU units. Next I want to see whether a single GELU unit per center, without tying, gets close to the RBF's 98.70.

I'm weighing whether the exact RBF is worth the extra complexity, since the gap to GELU is only about one sample in 1496. The tied two-unit exp approximation would reproduce the RBF KRR scores, so it stays on the table. I still need to check whether small λ makes the coefficients large enough to cause float32 precision trouble.

I'm checking the float32 error from summing 1496 weighted terms, estimating it at roughly 0.01, which is small but could flip a near-tie. Then I want to check how the decode step handles the LN2 shift when the output scale is large.

I'm looking at the pooler weights next, planning to overwrite rows 0 through 7 so they produce the centered output for each class. Then I want to settle whether to realize the exact RBF with tied GELU approximations of exp, or use the simpler untied GELU version, by measuring how well each one fits.

Next I'm getting the empirical distribution of u, so I can check how well a GELU combination matches exp(0.8u) over the range that actually occurs.



RBF kernel ridge regression scores 98.70 under 20-fold CV, versus 98.63 for the best untied GELU network. I'm checking whether a fixed tied GELU combination can approximate the exp(0.8u) term that makes up RBF's decision rule.



## tool_use: run_python

```json
{
 "code": "exec(open('krr.py').read())\nfrom scipy.special import erf\ndef gelu(a): return 0.5*a*(1+erf(a/np.sqrt(2)))\ny=np.load('base_feats.npz')['y']; X1=np.load('X1_full.npy'); F=X1[:,KEEP]\nZ,_=zs(F); d=F.shape[1]; G=Z@Z.T/d\nprint('u quantiles', np.quantile(G,[0,0.001,0.01,0.5,0.99,0.999,1]).round(3), 'diag', np.quantile(np.diag(G),[0,0.5,1]).round(3))\nu=np.linspace(G.min()-0.3, G.max()+0.3, 4000); tgt=np.exp(0.8*u)\nfrom itertools import product\nbest=[]\nfor g1_,b1_,g2_,b2_ in product([0.3,0.5,0.8,1.2],[-1,-0.5,0,0.5,1],[0.3,0.5,0.8,1.2,2],[-2,-1,-0.5,0,0.5,1]):\n    A=np.stack([np.ones_like(u), gelu(g1_*u+b1_), gelu(g2_*u+b2_)],1)\n    c,*_=np.linalg.lstsq(A,tgt,rcond=None); err=np.abs(A@c-tgt).max()/tgt.max()\n    best.append((err,(g1_,b1_,g2_,b2_),c))\nbest.sort(key=lambda t:t[0]); \nfor e,p,c in best[:5]: print(e,p,c.round(3))\nA=np.stack([np.ones_like(u), gelu(0.8*u+1)],1); c,*_=np.linalg.lstsq(A,tgt,rcond=None); print('1unit', np.abs(A@c-tgt).max()/tgt.max())\n"
}
```

