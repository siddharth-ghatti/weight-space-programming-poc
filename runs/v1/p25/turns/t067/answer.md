## reasoning summary

R2 sits at a median of 0.29 and stays there across layers, so the stash is carried through consistently, which is what an affine transform with a common offset should give. But R2 is already 0.29 at L1, and the reason is that the CLS constant term isn't proportional to the stash, so a regression on the stash alone with a common offset can't capture it.

The constant term in CLS is what breaks the fit: it isn't proportional to the stash, so a shared-offset regression on the stash alone can't absorb it. I'll add the CLS constant vector to the regressor and refit each sample. The α values shrink steadily across layers, so the offsets are doing most of the work, which makes the per-sample α spread look unreliable. The bigger worry is the L1 relative change in CLS, which is large enough that I need to check whether head 0 or the eight ablated dimensions caused it.

Now I'm going to redo the regression with the CLS constant vector included, since that term is what the stash-only fit misses. The per-sample α values shrink steadily across layers, so the offsets seem to dominate and the spread looks unreliable. The larger question is the big L1 change in CLS, and I need to work out whether head 0 or the eight ablated dimensions is responsible.

I'm checking the CV accuracy on the CLS and mean features at layer 9 along with the stash, and I'll revert the stash if accuracy drops below 97.64%. Before spending two passes on that, I'm weighing whether the design needs changing first, since the L1 relative change may be inflated by the small spread of CLS across sentences at that layer.

I'm weighing two possible refinements before the next full pass: shrinking the stash at layer 1 to reduce the disruption to the layer norm, and checking whether the stash signal at layer 9 is still large enough to matter against the common offset.

I'm checking the regression itself first, since the stash-only fit is what's leaving the CLS constant term unaccounted for.



R² is unchanged at 0.29 across layers, but my regression omitted the constant CLS term in the stash dims, so I'm redoing it with that term included. The CLS states moved about 25% of their spread at L9, which is only acceptable if the features stay equally predictive, so I'm checking the regression first.



## tool_use: run_python

```json
{
 "code": "import numpy as np, json\nF=np.load('feats.npz'); cfg=np.load('stash_cfg.npz'); S=cfg['S']; A0=cfg['A0']; b0=cfg['b0']\nM0=F['M0'][:64].astype(np.float64); st=M0@A0.T+b0\nh0=F['C0'][0,S].astype(np.float64); print('C0 const?', np.abs(F['C0'][:,S]-F['C0'][0,S]).max())\nfor l in [1,3,5,9]:\n    C=np.load(f'pass_11/cls_hidden_L{l}.npy').astype(np.float64)\n    al=[];r2=[];be=[]\n    for i in range(64):\n        z=4*st[i]+h0\n        X=np.c_[z,np.ones(8)]; coef=np.linalg.lstsq(X,C[i,S],rcond=None)[0]; al.append(coef[0]); be.append(coef[1])\n        pred=X@coef; r2.append(1-((C[i,S]-pred)**2).sum()/((C[i,S]-C[i,S].mean())**2).sum())\n    al=np.array(al)\n    print(f'L{l}: alpha {al.mean():.4f} cv {al.std()/al.mean():.3f}  R2 min {np.min(r2):.5f} beta std {np.std(be):.3f}')"
}
```

