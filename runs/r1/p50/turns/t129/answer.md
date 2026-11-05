## reasoning summary

I'm comparing the stress results across the configurations: m2048 comes out lower on rms than the m1536 runs. At k=2 the m2048 setup flips about 34 per thousand against 48 for m1536, so I'm considering how the 400-item test set might behave if its margins shrink from the practice margins by roughly that factor.

I'm working out what k should be for the test margins. Practice accuracy is 92.5% and test is 77.5%, and shrinking the logits wouldn't change practice accuracy, so the gap probably reflects test items being less confidently predicted. I'd need the average confidence on test to pin k down, and the earlier estimate of k around 1.5 to 2 is what I'm checking against.

I'm estimating how many flips the 400-item test set would produce by scaling the practice rates, and noticing that the older estimate of 8-11 doesn't match the numbers I get. The gap is about 12-19 flips, which would make the failure probability much higher than I'd like, so I want to rederive the scaling carefully and question whether a simple logit-shrinking model is even the right one for test items.

I'm now going to look at how the base margins are spread out, especially how many test items sit below roughly 0.25 to 0.5, since that is what decides how many could flip.



The older m2048 run is less noisy under stress, with 34 flips per 1000 at k=2 versus 48 for m1536 v1. Next I'll look at the practice margin distribution to calibrate the shrink factor k.



## tool_use: run_python

```json
{
 "code": "import json\nbl=np.load('base_logits.npy'); y=np.load('y.npy')\ndef sm(z): z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)\ns=np.sort(bl,1); marg=s[:,-1]-s[:,-2]\nprint('practice margin quantiles', np.quantile(marg,[.01,.02,.05,.1,.2,.5]).round(2))\nfor k in [1,1.5,2,2.5,3,4]:\n    p=sm(bl/k); conf=p.max(1)\n    # expected acc if calibrated\n    print(k, 'avg conf %.3f'%conf.mean(), 'frac margin<.3: %.3f'%((marg/k)<.3).mean(), ' <.5: %.3f'%((marg/k)<.5).mean())\nlg=np.load('lg_pass25.npy'); d=lg-bl\npb=bl.argmax(1)\ndm=(d[np.arange(1000),pb]-np.where(np.eye(4)[pb]>0,-np.inf,lg).max(1)+ np.where(np.eye(4)[pb]>0,-np.inf,bl).max(1))\nprint('margin change quantiles |dm|',np.quantile(np.abs(dm),[.5,.9,.95,.99]).round(3))\n"
}
```

