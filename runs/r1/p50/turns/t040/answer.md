## reasoning summary

Most units are sparsely active, but their contributions are spread out, so covering 95% of the variance takes roughly 2000 of them, and none are dead or purely linear that I could simply drop. With only about 5,000 tokens, each unit fires only a hundred or so times, so the rare ones that contribute heavily when they do fire might matter for specific tokens. Hmm, so I'm weighing how FFN pruning would handle that.

I'm weighing whether to accept limited FFN compression, since the units are too spread out to prune cleanly. The real question now is risk versus size: m=2560 looks safe at about 5.7% smaller, while m=2048 saves 11.5% but feels riskier.

I'm estimating test-time flip risk by treating test margins as practice margins scaled down by some factor s, then checking for each example whether its own noise would flip the decision. In-sample that gives about 1.6% at s=1, and I want to see how it grows for s=0.6 and s=0.4.



Units fire sparsely but contribute broadly (about 2,000 needed for 95% of variance), so narrowing the FFN is inherently lossy here. Next I'll estimate test flips by shrinking practice margins and applying the measured noise.



## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'common.py').read())\no = np.argsort(-BL,1); r=np.arange(1000); mb = BL[r,o[:,0]]-BL[r,o[:,1]]\nprint('base in-sample: frac margin<1', np.mean(mb<1), 'acc in margin<1', np.mean(bp[mb<1]==y[mb<1]), '2nd-choice correct', np.mean(o[mb<1,1]==y[mb<1]))\nfor nm,pth in [('drop4','pass_3'),('A L0-4 m1024','pass_11'),('m1536','pass_10'),('m2048','pass_12')]:\n    lg = np.load(WORK/pth/'logits.npy')\n    dm = (lg[r,o[:,0]]-lg[r,o[:,1]])-mb\n    # flips w.r.t. top-2 only (approx)\n    res = [f\"s={s}: {np.mean(s*mb+dm<0)*100:.1f}%\" for s in [1,0.6,0.4,0.25]]\n    print(nm, ' '.join(res))"
}
```

