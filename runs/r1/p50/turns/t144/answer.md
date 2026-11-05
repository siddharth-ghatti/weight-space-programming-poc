## reasoning summary

The Gaussian model predicts far fewer flips than what I observed in the empirical-noise sim, so the real noise must be heavier-tailed or structured rather than independent per logit. I'm wondering whether the noise lines up with the margin direction, or whether it concentrates on items with low margins, since those sit near the decision boundary where the representation is more sensitive.

The empirical sim flips about 1.6 times as often as the Gaussian model predicts, so the noise may be shrinking logits toward the mean, which would shrink margins across the board. I'll check whether the noise correlates with the centered base logits, and whether a systematic shrinkage term could explain the gap.

I'm checking whether the mean margin change is systematically negative, especially among low-margin items, since that would make flips more likely. The base-right items seem to lose margin while base-wrong ones gain against gold, so I'll look at how the mean shift depends on margin. The open question is whether that regression toward the mean, which pruning would produce on the memorized training items, carries over to test items, where nothing was memorized.

I'm going to see how the mean margin change depends on base margin, specifically for the m1536v1 run, since that would show whether pruning systematically pushes margins down.



The Gaussian model undershoots the observed flips (4.7 vs 7.6 at k=1), so the real noise likely shrinks margins non-uniformly. I'll check the mean signed margin change against base margin, split by whether the base prediction was correct.



## tool_use: run_python

```json
{
 "code": "import json\nbl=np.load('base_logits.npy'); y=np.load('y.npy')\npb=bl.argmax(1); s=np.sort(bl,1); marg=s[:,-1]-s[:,-2]; ru=np.argsort(bl,1)[:,-2]\ndef dmarg(lg):\n    # change in (base-pred logit - max other logit)\n    oth=np.where(np.eye(4)[pb]>0,-np.inf,lg).max(1)\n    return (lg[np.arange(1000),pb]-oth)-marg\nfor f in ['lg_pass25.npy','lg_pass31_m1792v2.npy','lg_pass30_m2048v2.npy']:\n    lg=np.load(f); dm=dmarg(lg)\n    print(f,'mean dm all %.3f'%dm.mean())\n    for a,b in [(0,.5),(.5,1),(1,2),(2,4),(4,99)]:\n        m=(marg>=a)&(marg<b); r=m&(pb==y); w=m&(pb!=y)\n        print('  marg %.1f-%.1f n=%3d mean dm %.3f sd %.3f | base right n=%3d %.3f | base wrong n=%2d %.3f'%(a,b,m.sum(),dm[m].mean(),dm[m].std(),r.sum(),dm[r].mean() if r.sum() else 0,w.sum(),dm[w].mean() if w.sum() else 0))\n"
}
```

