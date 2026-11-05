## reasoning summary

This is a pessimistic result: with 50% and 33% rates and ten items, the failure chance is about 16%, so accuracy is the real constraint and F1 comes second. Eight items would bring the risk down to about 10%, which means the test set needs to be around five or six items for under 5% risk. The open question is what N_test is for our models, so I'll try to pin that down from the de-biased noise estimate.

I'm now setting up a per-item estimate of how the pairwise margin shifts, de-biased by removing the group-dependent mean shift, so that the noise is roughly symmetric. From there I want to turn that noise distribution into a flip probability for each practice item, and scale the result by an assumed test-to-practice margin density to estimate N_test for a few values of k.



The simulation puts failure risk near 5% at about 6 test flips and 24% at 13, with accuracy the binding bar. I'm now estimating the expected test flips per model using de-biased noise scaled to 400 test items.



## tool_use: run_python

```json
{
 "code": "exec(open('common.py').read())\ns=np.argsort(-BL,1); t1=s[:,0]; t2=s[:,1]\nmarg=BL[np.arange(1000),t1]-BL[np.arange(1000),t2]; right=bp==y\ndef debiased_flips(lg):\n    # noise on runner-up pairwise margin\n    dm=(lg[np.arange(1000),t1]-lg[np.arange(1000),t2])-marg\n    eps=dm.copy(); eps[right]-=dm[right].mean(); eps[~right]-=dm[~right].mean()\n    # expected flips: P(eps < -mu) using empirical eps distribution (ignore self)\n    es=np.sort(eps)\n    pf=np.searchsorted(es, -marg)/len(es)\n    return pf.sum(), eps.std(), (lg.argmax(1)!=bp).sum()\nfor name in ['lg_pass25.npy','lg_pass26.npy','lg_D1_L01.npy','lg_D2_L234.npy']:\n    E,sd,obs=debiased_flips(np.load(name))\n    print(f'{name}: observed flips {obs}, de-biased expected practice flips {E:.1f}, eps std {sd:.3f}; test N for k=1,1.5,2,3: '+', '.join(f'{0.4*k*E:.1f}' for k in [1,1.5,2,3]))\n"
}
```

