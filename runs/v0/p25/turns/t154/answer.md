## reasoning summary

Both runs landed at 0.994, so the random and piv variants are equal here, even though the piv one had some numerical trouble. Using all points as centers would reach 0.9951, but with only the 1197 practice centers that totals 3143, over the 3072 cap, so it can't happen in the final setup. Maybe the center set doesn't have to be drawn only from the core points: any 3072 of the 3442 could serve, and the CV practice-train centers come to 1197 plus my 1576, or 2773 in total.

I'm weighing whether the practice centers could be traded for more of my own, though that probably makes little difference. I'm also considering using the pooler's 768 tanh units as extra random features on the normalized CLS output, with the classifier fit by ridge over those plus the 8 logit units.

I'm considering whether the pooler's 768 tanh units could serve as extra random features, with the 760 unused dimensions of the normalized CLS output feeding new kernel centers. The scaling of that normalized output varies per sample, which complicates things, so the gain is uncertain.

I'm working out how the joint fit would look: the 3072 gelu features and 760 tanh features would share one ridge regression, giving separate output coefficients for each set. The open question is whether the FFN output can carry those coefficients, or whether the pooler units 0–7 have to read them directly.

I'm checking whether the pooler's tanh units can take the 760 extra features, using a small scale so the logit units stay in their roughly linear range. The open issue is that per-sample LayerNorm scaling makes the normalized component vary between inputs, which could break the linear approximation.

I'm checking whether the pooler's tanh units can carry 760 extra features while keeping units 0–7 in their near-linear range. The LayerNorm scale looks almost constant when the logit component is small, so the normalized output should stay close to what the linear approximation needs.

I'm checking whether the variance along the lowest-variance directions of the normalized output is small enough to ignore at a scale of 1. Raising the scale to 10 would make the per-sample normalization factor vary by a few percent, which would slightly distort the tanh feature values, though the mapping would still be deterministic.

I'm checking whether the lowest-variance directions of the normalized output add enough noise to matter at S=1, and I'm weighing that against a 760-center expansion, which would take the total from 3072 to 3832 and might lift accuracy from about 0.994 toward 0.995.

I'm weighing whether a few more of my own sentences would help more than the marginal gain from extra centers, since the CV is already near 0.994 and the remaining errors look largely label-ambiguous. A batch of roughly 460 more sentences might add about 0.2%, which is modest but real, and it avoids the complexity of the joint pooler construction.

I'm planning to train the next version on all 1946 of my own sentences, then verify it in one pass. After that I might write about 460 more sentences, rebuild, and check again. I also want to re-tune the regularization and weighting for the new setup, starting with a quick random selection.



Random and pivoted-Cholesky selection both score 0.994, so the selection strategy makes no difference. Next I'll re-tune the weight and lambda for this setup, which uses 3442 training points and 1576 centers, before building v5.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0,'.')\nfrom kh import *\nD=load(); d5=np.load('my5_feats_clean.npz')\nZp,yp=D['Zp'],D['yp']\nZmine=np.concatenate([D['Zm'],D['Z3'],D['Z4'],d5['Z']]); ymine=np.concatenate([D['ym'],D['y3'],D['y4'],d5['y']])\nNC=1576\ndef cv_cfg(wx, lam, seeds=(0,1,2), nf=5):\n    accs=[]\n    for s in seeds:\n        fo=folds(yp,s,nf); pred=np.zeros(1496,int)\n        sel=np.random.default_rng(100+s).permutation(len(Zmine))[:NC]\n        for f in range(nf):\n            tr=fo!=f\n            Cc=np.concatenate([Zp[tr],Zmine[sel]])\n            Zt=np.concatenate([Zp[tr],Zmine]); yt=np.concatenate([yp[tr],ymine]); wt=np.concatenate([np.ones(tr.sum()),np.full(len(Zmine),wx)])\n            a,b=fit(gelu(Zt@Cc.T/768),yt,wt,lam)\n            pred[~tr]=np.argmax(gelu(Zp[~tr]@Cc.T/768)@a+b,1)\n        accs.append((pred==yp).mean())\n    return round(np.mean(accs),4), np.round(accs,4)\nfor wx,lam in [(0.25,0.005),(0.4,0.003),(0.4,0.012),(0.6,0.007),(0.8,0.007)]:\n    print(wx,lam,cv_cfg(wx,lam))\n"
}
```

