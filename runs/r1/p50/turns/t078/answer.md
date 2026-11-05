## reasoning summary

I'm weighing the risk, which depends on the test margin distribution I don't know. Under a 0.7 shrink, which seems plausible, m1536v2 fails about 15% of the time and m2048old about 4.6%, and under 0.5 both look risky. I also need to question whether noise stays independent of margin, since test items might be less typical and carry more noise than the in-sample case the simulation assumes.

I'm checking whether the simulation's random sign on flips is fair for test items, since in-sample flips on base-correct items may be biased by memorization. The test margins will probably sit below the in-sample median of 2.95, so a 0.7 shrink looks plausible and 0.5 looks pessimistic. I'm also wondering whether a cross-validated head refit could estimate the test margin distribution, though the encoder has already memorized all items, so those margins would still be in-sample.

I'm now testing a shift model for the test margins: subtract a constant from the in-sample signed margins until about 22.5% turn negative, instead of assuming a uniform scale. Then I'd check how many items sit near zero and simulate flips across that boundary to see the net effect on each model.

I'm considering a noise model instead of a shift: treating test margins as the in-sample signed margins plus random Gaussian noise, with the noise scale chosen so that about 22.5% come out negative. That feels like a reasonable picture of how fine-tuning overfit might show up on unseen items, so I'll compute both the shift and noise versions to compare.



Test-margin risk hinges on an unknown distribution, so I'm modelling two variants (a downward shift or Gaussian blur, calibrated to 22.5% base error) and simulating their flips.



## tool_use: run_python

```json
{
 "code": "exec(open('common.py').read())\nidx=np.arange(1000)\noth = BL.copy(); oth[idx,y] = -np.inf\nsm = BL[idx,y] - oth.max(1)\nprint('in-sample signed margin <0:', np.mean(sm<0))\n# calibrate shift\nfrom scipy.optimize import brentq\ndelta = brentq(lambda d: np.mean(sm-d<0)-0.225, 0, 5); print('shift delta', delta)\nrng=np.random.default_rng(1)\ndef frac_neg_tau(t):\n    return np.mean((sm[None,:]+t*np.random.default_rng(5).standard_normal((200,1000)))<0)\ntau = brentq(lambda t: frac_neg_tau(t)-0.225, 0.01, 10); print('tau', tau)\ndef noise(L):\n    # noise on (true - best other) direction per item using item's own runner-up: approximate via base top/runner-up margin noise\n    dm=L-BL; top=bp; ru=np.argsort(BL,1)[:,-2]\n    return dm[idx,top]-dm[idx,ru]\nvariants = {'m2048old':np.load('pass_12/logits.npy'), 'm1536v2':np.load('lg_m1536_v2.npy'), 'm1536old':np.load('pass_10/logits.npy')}\nfor model in ['shift','tau']:\n    for name,L in variants.items():\n        nm = noise(L); losses=[]\n        for t in range(3000):\n            j = rng.integers(0,1000,400)\n            s_t = sm[j]-delta if model=='shift' else sm[j]+tau*rng.standard_normal(400)\n            n = rng.choice(nm,400)*rng.choice([-1,1],400)\n            new = s_t + n\n            broken = ((s_t>0)&(new<0)).sum(); fixed = ((s_t<0)&(new>0)).sum()\n            losses.append(broken-fixed)\n        losses=np.array(losses)\n        print(f'{model} {name}: mean {losses.mean():.2f} sd {losses.std():.2f} P(loss>4) {np.mean(losses>4):.3f}  P(>3) {np.mean(losses>3):.3f}')"
}
```

