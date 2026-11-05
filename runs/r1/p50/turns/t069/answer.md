## reasoning summary

Pruning layer 5 down to 1024 hardly costs anything, and the layers 0–4 are the real bottleneck at that width. Correction at layer 5 doesn't help. At 1024 the six-layer model reaches about 0.964 agreement and 0.892 in-sample accuracy, which is probably too lossy; the older 1536 run did better, and 2048 better still. I'm wondering whether to test a different width.

I'm working out how the in-sample flips would carry over to the test set. At m=1024, accuracy drops 3.3 points with about 36 flips, and nearly all of them hurt. I'm thinking about splitting the flips into fixed and broken, and whether low-margin items are more likely to be base-wrong on test.

I'm estimating how the in-sample flips would carry over to the test set, using a simple net-loss model: a flip costs one point if the base model was right, and gains one point only sometimes when the base was wrong. Test items are base-wrong about 22.5% of the time, and low-margin items are probably less accurate on test than in-sample, so the net loss there could be meaningful.

I'm going to estimate the test margin distribution the same way I did in-sample, starting with the base model's margin fractions below 0.5, 1, and 2. The noise is treated as roughly independent of margin, so flip probability per margin bin can come from the in-sample data and then get applied to an assumed test distribution. Since the base is 77.5% accurate on test, I'll anchor that assumed distribution to it rather than to the in-sample numbers.

I'm estimating test flip counts by scaling the in-sample rate by the ratio of low-margin mass, since test has about three times as many base errors. That puts m=1024 at roughly 28 to 40 flips and a net loss near 8 points, well over the 4-point budget. m=1536 looks borderline, and m=2048 is borderline too, so even the wider models may not clear the bar.

I'm going to check the in-sample flips against the base model's correctness to see how often a flip actually fixes an error versus breaks a correct prediction. Comparing the m=1536, m=2048, and m=1024 runs should show whether the net-loss estimate holds up or whether flips are closer to neutral on test.



Pruning layer 5 to m=1024 barely moves agreement, so the errors come from layers 0–4, and the in-sample accuracy of 0.892 looks too low. I'm building a risk model of flip probability against base margin to judge how much in-sample agreement I need.



## tool_use: run_python

```json
{
 "code": "exec(open('common.py').read())\nimport json\ns = np.sort(BL,1); marg = s[:,-1]-s[:,-2]\ncands = {}\nfor d in ['pass_10','pass_12']:\n    m = json.load(open(d+'/meta.json'))\n    try: lg = np.load(d+'/logits.npy')\n    except Exception as e:\n        print(d, os.listdir(d)); continue\n    cands[d]=lg\nprint({k:v.shape for k,v in cands.items()})\nfor k,lg in cands.items():\n    p = lg.argmax(1); fl = p!=bp\n    bc = bp==y\n    print(k, 'flips', fl.sum(), 'base-correct broken', (fl&bc).sum(), 'base-wrong fixed', (fl&~bc&(p==y)).sum(), 'base-wrong->other wrong', (fl&~bc&(p!=y)).sum())\n    dm = lg - BL\n    # margin noise: change of (top - runner-up) of base classes\n    top = bp; ru = np.argsort(BL,1)[:,-2]\n    nm = (dm[np.arange(1000),top]-dm[np.arange(1000),ru])\n    print('  noise on margin: std', nm.std(), 'mean', nm.mean(), 'mean|.|', np.abs(nm).mean())\n    for lo,hi in [(0,0.5),(0.5,1),(1,2),(2,4),(4,99)]:\n        mk = (marg>=lo)&(marg<hi); print(f'   margin {lo}-{hi}: n {mk.sum()} flips {fl[mk].sum()} base acc {bc[mk].mean():.2f}')"
}
```

