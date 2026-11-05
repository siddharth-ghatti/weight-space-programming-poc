## reasoning summary

The E[da] shift for the m1536v1 model at k=1.25 is much more negative than for m2048, so the mean shift looks like it matters a lot here. Under calibration, that mean is being driven by the noise flipping items where the base's top probability is above the new class probability. I also need to check a subtlety: the simulation draws its noise from the practice setting, and the memorization finding says that the edit lowers the margin on base-right items while raising it on base-wrong ones.

I'm checking whether the simulation's noise picks up the memorization effect, and it doesn't, since the sim samples gold from the base softmax rather than the practice labels. Under linear scoring, m1536 still comes out ahead of m2048 on expected value, and m1792 lands in between.

I'm wondering whether the heavy-tailed noise comes from a handful of specific tokens with outlier activations, which would mean robust measures could help more than a bigger model would.

I'm weighing whether to probe the token-level hidden-state errors on the high-noise items at each layer, which would take two passes, or to simply fold those items' tokens into the fitting data, since they're short and allowed in practice.

I'm wondering whether the fit data, which was mostly the longest sequences, underrepresents the token types in the short high-noise items. If so, the compensation would be weaker on those tokens, and adding them to the fitting set could help, though it would make the practice evaluation optimistic. I'd want to check whether short items really have higher noise before deciding.

I'm checking whether the 46-token item's noise is an outlier, about four times what its length predicts. To see if the heavy tail is real, I want the normalized noise for every item, and then the texts of the largest ones to see what they have in common.



The m1536 flips average about −1.9 in expected net change, and their fail risk is 10–22% against 6–9% for m2048 v2. I'm checking whether the heavy tail comes from a few identifiable inputs by normalising noise by sqrt(length) and looking at the outliers.



## tool_use: run_python

```json
{
 "code": "import json\nbl=np.load('base_logits.npy'); ids=json.load(open('all_ids.json'))\nlj=json.load(open('lens.json')); lm=dict(zip(lj['ids'],lj['lens'])); Ln=np.array([lm[i] for i in ids])\npr={r['id']:r for r in practice()}\nfor f in ['lg_pass25.npy','lg_pass30_m2048v2.npy']:\n    lg=np.load(f); d=lg-bl; ri=np.sqrt((d**2).mean(1)); z=ri*np.sqrt(Ln/100)\n    print(f,'normalized noise quantiles',np.quantile(z,[.5,.9,.95,.99,1]).round(3))\nlg=np.load('lg_pass25.npy'); d=lg-bl; ri=np.sqrt((d**2).mean(1)); z=ri*np.sqrt(Ln/100)\nlg2=np.load('lg_pass30_m2048v2.npy'); z2=np.sqrt(((lg2-bl)**2).mean(1))*np.sqrt(Ln/100)\nprint('corr of item noise between m1536 and m2048', np.corrcoef(z,z2)[0,1].round(3))\no=np.argsort(-z)[:8]\nfor i in o:\n    print(ids[i], Ln[i], 'z=%.2f z2=%.2f'%(z[i],z2[i]), pr[ids[i]]['label'], repr(pr[ids[i]]['text'][:160]))\n"
}
```

