## reasoning summary

I'm checking whether the short sequences, under 70 tokens and about a quarter of the items, are driving the error: they have higher residuals and account for 10 of the 19 flips. I suspect a distribution shift, since our fit used only the longest sequences, so the short ones may differ in position effects and in the share of CLS and end-of-sequence tokens.

The short sequences look like they account for much of the error, but part of that is just the expected effect of sequence length on noise. Scaling the rms from length 100 down to 45 predicts about 0.22, against the observed 0.27 for sequences under 50 tokens, so the length effect explains most of the gap and a smaller extra may remain. Weighting the fit per sequence rather than per token might match the objective better, since sequence-level error scales roughly as the inverse of length, which would push weight toward the short sequences. The complication is that the fit data only includes sequences of 120 tokens or more, so short-sequence data would have to be added, and that is limited by the cap of 32 sequences per pass.

Checking whether the short sequences are simply noisier by construction: their per-token error scales roughly with 1/√length, which accounts for most of the gap, leaving a small residual excess. Position-weighting the fit might help the short-sequence distribution, but the core noise is just token error averaging, so the real lever is reducing token error overall.

I'm weighing whether to stop pursuing the compensation approach, since it looks saturated, and to settle on a decision. Next I want to recheck the risk model using the calibration-derived k≈1.5, by simulating synthetic test sets from practice items with shrunk logits and noise from the edited model, which should reproduce the observed test accuracy.

I'm checking the failure threshold against the base test score, where a net loss of more than 4 items or a macro-F1 drop over 0.02 counts as failing. Then I want to run the simulation with expected values instead of sampled gold labels: for each item, a change in correctness only happens where the edited and base predictions differ, weighted by the calibrated probabilities of each prediction being the gold label.

I'm setting up a Monte Carlo check of the failure risk: resampling 400 of the 1000 practice items with replacement, drawing gold labels from the calibrated probabilities, and tallying the net change in correct answers and the macro-F1 shift. I'll run it for each candidate version and repeat at a couple of calibration temperatures to see how sensitive the result is to overconfidence.



Short paragraphs have the most noise, and that error averages out rather than being a bias I can fix, so the remaining choice is m. I'm now running a Monte Carlo simulation of the test set to estimate the risk of failing, sampling 400 items and checking both the accuracy and macro-F1 bars.



## tool_use: run_python

```json
{
 "code": "import json\nbl=np.load('base_logits.npy'); ids=json.load(open('all_ids.json'))\ndef sm(z): z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)\ndef f1macro(g,p):\n    f=[]\n    for c in range(4):\n        tp=((p==c)&(g==c)).sum(); fp=((p==c)&(g!=c)).sum(); fn=((p!=c)&(g==c)).sum()\n        f.append(2*tp/max(2*tp+fp+fn,1))\n    return np.mean(f)\ndef load_pass(p):\n    m=json.load(open(p+'/meta.json')); lg=np.load(p+'/logits.npy'); mid=m['ids']\n    return lg[[mid.index(i) for i in ids]] if mid!=ids else lg\nmodels={'m1536v1':np.load('lg_pass25.npy'),'m1536v3':np.load('lg_pass29.npy'),'m2048old':load_pass('pass_12'),'m1536old':load_pass('pass_10'),'m1024zero':load_pass('pass_11')}\nrng=np.random.default_rng(1); R=4000; N=400\nfor k in [1.0,1.25,1.5,2.0]:\n    P=sm(bl/k); cum=P.cumsum(1)\n    res={}\n    idxs=rng.integers(0,1000,(R,N)); u=rng.random((R,N))\n    golds=(u[...,None]>cum[idxs]).sum(-1)\n    bp=bl.argmax(1)\n    for name,lg in models.items():\n        d=lg-bl; ep=(bl/k+d).argmax(1)\n        fails=0; accfail=0; f1fail=0; flips=[]\n        for r in range(R):\n            ii=idxs[r]; g=golds[r]; b=bp[ii]; e=ep[ii]\n            da=(e==g).sum()-(b==g).sum(); flips.append((e!=b).sum())\n            fa=da<-4; ff=f1macro(g,e)<f1macro(g,b)-0.02\n            accfail+=fa; f1fail+=ff; fails+=(fa or ff)\n        res[name]=(np.mean(flips).round(1), accfail/R, f1fail/R, fails/R)\n    print('k',k,'base acc %.3f'%P.max(1).mean())\n    for n,v in res.items(): print('   %-10s flips %.1f  P(accfail) %.3f P(f1fail) %.3f P(fail) %.3f'%((n,)+v))\n"
}
```

