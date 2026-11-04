## reasoning summary

I'm weighing whether to add more sentences before building v4, since there are 86 spare slots out of 2986 centers. One idea is to keep the extra sentences as training points rather than centers, and I'd test that by comparing centers taken from a subset against using all of them.

I'm now checking whether the layer 10 feed-forward block could be repurposed for more features, but it feeds the layer 11 features I'm already using, so I can't reuse it. The layer 11 width of 3072 remains the real limit. Next I'll build v4 with the 2986 centers and check it with a predict pass on practice ids.

I'm checking that the Z3 and Z4 centers were standardized with the kernel_v3 mean and scale, and whether nbar still matches mu. I'm also making sure the U directions stay as they are. Next I want to look at the magnitudes of A, since more centers and a lower lambda could make the coefficients large enough to cause precision problems in the FFN output.



The v4 structure checks out, so only the new FFN centers and W2/b2 change. I'm fitting on all 2986 points with the chosen weights and lambda, and checking for coefficient blow-up in fp32.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom kh import *\nD=load()\nk=np.load('kernel_v3.npz'); mu,sd,nbar=k['mu'],k['sd'],k['nbar']\nprint('nbar vs mu', np.abs(nbar-mu).max())\nC=np.concatenate([D['Zp'],D['Zm'],D['Z3'],D['Z4']]).astype(np.float64)\nyc=np.concatenate([D['yp'],D['ym'],D['y3'],D['y4']])\nw=np.concatenate([np.ones(1496),np.full(len(C)-1496,0.4)])\nlam=0.007\nP=gelu(C@C.T/768); A,b0=fit(P,yc,w,lam)\nF=P@A+b0\nprint(len(C), np.abs(A).max(), np.linalg.norm(A), 'train acc', (F.argmax(1)==yc).mean(), 'prac',(F[:1496].argmax(1)==yc[:1496]).mean())\n# fp32 sensitivity: perturb Z by fp32 rounding-ish noise 1e-4 relative\nrng=np.random.default_rng(0); Cn=C+rng.normal(0,1e-3,C.shape)\nFn=gelu(Cn@C.T/768)@A+b0; print('logit change w/ 1e-3 noise', np.abs(Fn-F).max())\nnp.savez('kernel_v4.npz',mu=mu,sd=sd,C=C,yc=yc,w=w,lam=lam,A=A,b0=b0,F=F,nbar=nbar)\n"
}
```

