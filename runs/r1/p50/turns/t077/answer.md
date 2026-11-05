## reasoning summary

The m1536 version comes out slightly better, with agreement at .981 and noise std down to .352 from .373. The older m2048 setup had a much lower std of .23, so noise rises noticeably as m shrinks. I'm wondering where that noise is coming from layer by layer, and whether an earlier analysis from pass 11 could show which layer contributes the most.

I'm looking at which layers push the most error forward, since early-layer mistakes carry through everything after them, and the per-layer relative errors at m=1024 run from .14 at L4 up to .27 at L1. Ranking units by their downstream effect on the final logits would be the more useful criterion, but a Fisher-style estimate needs gradients I can't use here, and finite differences through forward passes look too costly.

I'm weighing whether a bigger configuration like m=1792 or m=2048 would cut the noise enough to justify the extra parameters, since each added step costs about 3% of the total. The scoring favors fewer exported parameters among passing models, and the gap between roughly 68M and 73M is only about 6%. I want to estimate the risk more carefully with a simulated test-margin distribution, but the probe's logit scale doesn't match the base model's, so that proxy needs care.

I'm weighing the test risk through a rough simulation, treating the test margins as a mix: wrong items drawn from the in-sample margins of base-wrong examples, and correct ones from base-correct margins scaled down, since memorized items overstate confidence. I'll resample the empirical noise for the m2048 setup and count how often flips fix or break predictions.

I'm going to check the m1024 noise from the L5 sim, then run a quick simulation of test flips using the noise std across the configurations.



The m=1536 build now matches at .981 against the old build's .977, with slightly lower margin noise. Next I'll simulate test outcomes to estimate the risk of a net loss before choosing m.



## tool_use: run_python

```json
{
 "code": "exec(open('common.py').read())\ns = np.sort(BL,1); marg = s[:,-1]-s[:,-2]; top=bp; ru=np.argsort(BL,1)[:,-2]\nbc = bp==y\nprint('margins wrong items: pct', np.percentile(marg[~bc],[10,25,50,75,90]).round(2))\nprint('margins right items: pct', np.percentile(marg[bc],[10,25,50,75,90]).round(2))\ndef noise(L):\n    dm=L-BL; return dm[np.arange(1000),top]-dm[np.arange(1000),ru]\nvariants = {'m2048old':np.load('pass_12/logits.npy'), 'm1536v2':np.load('lg_m1536_v2.npy'), 'm1536old':np.load('pass_10/logits.npy')}\nrng = np.random.default_rng(0)\nfor shrink in [1.0, 0.7, 0.5]:\n    for name,L in variants.items():\n        nm = noise(L)\n        losses=[]\n        for t in range(4000):\n            nw = rng.binomial(400, 0.225)\n            mw = rng.choice(marg[~bc], nw); mr = rng.choice(marg[bc], 400-nw)*shrink\n            # noise drawn independently (sign random: noise sign relative to margin)\n            n_w = rng.choice(nm, nw)*rng.choice([-1,1],nw); n_r = rng.choice(nm, 400-nw)*rng.choice([-1,1],400-nw)\n            fw = (n_w < -mw); fr = (n_r < -mr)\n            fixed = rng.binomial(fw.sum(), 0.6)\n            losses.append(fr.sum() - fixed)\n        losses=np.array(losses)\n        print(f'shrink {shrink} {name}: mean net loss {losses.mean():.2f}, sd {losses.std():.2f}, P(loss>4) {np.mean(losses>4):.3f}')"
}
```

