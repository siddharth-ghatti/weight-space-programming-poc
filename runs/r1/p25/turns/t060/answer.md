## reasoning summary

The new model at m=2048 agrees with the original head 98.5% of the time, with logit R² 0.996 and 94% agreement on low-margin cases, a clear jump over the m=1536 run at 92.2%. Fixing the CLS token, which had been acting as an attention sink, seems to be what made the difference. Practice accuracy is 0.916 against 0.925 for the base, with 15 disagreements out of 1000, and I'd expect the harder test set to show around 3.

I'm weighing two paths now: pushing to a smaller m like 1536, which should land near 0.96 agreement, or tightening fidelity at m=2048 with more data first. The parameter savings look worth it, roughly 4.7M from 2048 down to 1536, though I'm wary that dropping whole layers would be too lossy.

I'm now working out a correction for the FFN: rather than fitting to the base FFN applied to the edited model's inputs, I'd target the pre-LayerNorm output the base would produce, so upstream errors get compensated. Since LayerNorm is scale and shift invariant, the target becomes the base's residual sum minus the edited model's attention output, using probe activations from the same 32 ids. I'm about to fit the FFN to that target.

I'm going to correct the layer-5 CLS fit by using the target a_base + f_base(a_base) - a_e, where a_e comes from one pass with 1000 ids, so that upstream drift gets compensated there too.

I'm weighing whether to spend the remaining budget on going down to m=1536 with corrections, or on locking in m=2048. Going lower would first need a measurement pass to see where the agreement actually lands before committing to the more expensive sequential correction.

I'm looking at where the errors come from at each layer. Local error at L0 is the main contributor, with R² staying roughly flat from L3 onward, so the compounding seems to stop after L2. More data for L0 through L2 might bring the error down a bit, though probably only modestly.

I'm weighing how to weight the errors: since the per-column ridge solve is independent for each output dimension, reweighting doesn't change the refit itself, only which dimensions are worth keeping during selection.

Checking whether m=1536 is viable: the CV errors roughly double at layers 0 through 2 (for example, L0 goes from 0.042 to 0.091), which I estimate would put agreement around 97.5 to 98%, with logit R² near 0.992. Going down to m=1024 looks too lossy, since those layer errors would be about 0.2. The scoring rewards fewer parameters among passing models, but a failure is costly, so I need to be sure the test set won't push the disagreement rate past the threshold.

Now I'm working out how practice disagreements might carry over to the test set, since the base model is probably more confident on training items than on test. The low-margin quartile looks like the better proxy, and it's already at 0.94 agreement for m=2048. Without test margins, I'm thinking of estimating flip probability as a function of base margin from the distribution of logit differences on practice.

I'm worried the flip rate on test could be much higher than on practice, since test items probably sit closer to the decision boundary. If about 6% of the bottom-quartile items flip at m=2048, that could mean roughly 24 flips on 400 test items, and with the base only slightly more often right on those, the net accuracy change could be enough to fail.

I'm checking how the 15 practice flips split by accuracy: base accuracy drops from 0.925 to 0.916, which means about 9 net losses. Since the base is fit to the labels, that number is probably optimistic, so I want to see how margin relates to accuracy on practice before estimating what the test set will do.

I'm now estimating the logit error spread at m=2048, comparing the std of the class-difference errors against the typical base margins. The governance class is small, so I'm also weighing whether the macro-F1 drop could be affected by how the test set is distributed.

I'm checking whether the per-class logit errors at m=2048 are small enough relative to the typical base margins to keep flips rare on the harder test set.



Protecting <s> with an LS refit lifted agreement with base to 0.985 and logit R² to 0.996. Next I'll check how the logit perturbation compares to base margins to estimate flips on the harder test set.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nlgb=np.load('pass_1/logits.npy'); y=np.load('y.npy')\nfor pd in ['pass_11','pass_12']:\n    lge=np.load(f'{pd}/logits.npy')\n    d=lge-lgb\n    bp=lgb.argmax(1)\n    srt=np.sort(lgb,1); marg=srt[:,-1]-srt[:,-2]\n    # perturbation of margin between top and each other class\n    pert=[]\n    for i in range(len(lgb)):\n        c=bp[i]; others=[j for j in range(4) if j!=c]\n        pert.append(np.max(np.abs(d[i,others]-d[i,c])))\n    pert=np.array(pert)\n    print(pd,'logit diff std',d.std(0).round(3),' pairwise pert median',np.median(pert).round(3),'p90',np.percentile(pert,90).round(3))\nprint('margin percentiles',np.percentile(marg,[5,10,25,50]).round(2))\n# accuracy by margin bin\nfor lo,hi in [(0,1),(1,2),(2,4),(4,8),(8,100)]:\n    s=(marg>=lo)&(marg<hi); print(lo,hi,s.sum(), (bp[s]==y[s]).mean().round(3) if s.sum() else '')\n"
}
```

