## reasoning summary

I'm weighing the honest numbers for m=1792 upstream: the full L5 FFN ceiling sits at .983 with 17 flips, and the base-only fit at .981 with 19 flips. The .985 from the earlier pass looks like in-sample overfitting, since the mixed fit used all 1000 items. That also means the dm std drops from about .29 to .21 only in-sample, which I'd attribute to overfitting a 768x1792 ridge regression.

I'm now checking the m=1792 numbers against the old m=2048 run: the honest 1792 fit gives 19 flips and a dm std of .29, which is worse than the 15 flips and .24 from the earlier run. An honest m=1536 would likely be worse still.

I'm checking whether the base-only L5 fit in the m=2048 run was itself overfitting, and comparing its edited-input agreement to the full FFN; the earlier CV suggests it wasn't much. I'm also noting that the L0-L2 fits, which used 64 items' tokens, hold out about as well as they fit.

I'm checking the upstream options next: sequential correction for L1 through L4, where mixed fits might generalize better given more tokens per parameter. The L5 result already warns that fixing upstream error downstream can raise in-sample fit while lowering held-out agreement.

I'm considering whether the upstream layers are where most of the remaining error comes from. One way to check is a diagnostic: keep the m=3072 structure, but zero out the pruned columns only in specific layers, so I can see how much each layer's pruning costs.

I'm considering whether a uniform m is the real constraint: L5's FFN only matters at the CLS token, so it could be pruned much harder, but the shared width forbids that. Dropping whole layers already hurt, so the realistic choice is between m=1792 (70.5M params), m=2048 (72.9M), or an aggressive m=1536 (68.1M).

I'm estimating the test-set flip counts for the three widths, roughly 10.5 for m=2048, 13.3 for m=1792, and 17 for m=1536. Using a per-flip net loss of about -0.2 to -0.3 with a variance near 0.85, I'm checking how often each option would fail.

I'm checking the risk numbers: a flip-based loss estimate puts the failure chance for the 2048 option at roughly 21-33% and for 1792 at roughly 30-44%. That seems high enough to make me question whether the base model is already borderline, and whether the per-flip loss of about -0.2 is realistic on the test set when the practice data shows about -0.77.

I'm estimating the per-flip net loss on test at around -0.15, since the practice numbers are probably inflated by in-sample labels. With that, the 2048 option comes out near a 17% chance of failing.

I'm weighing whether to go with m=1792 or m=2048, and reckon the risk of failing is roughly 17-23% for each. Dropping to m=1536 pushes the risk higher still, while m=2560 would cut the expected flips a lot but would give up most of the size savings. I'm inclined to accept a moderate reduction with a high chance of passing, since the score function is unclear.

I'm looking at whether upstream fidelity is the lever: L1 has the largest CV FFN error at m=1792, around .080, versus about .05 for L0 and L2, and L3 and L4 are negligible. I'm wondering what makes L1's FFN harder to approximate with fewer units.

I'm now looking at whether discarding units can be compensated by merging them into kept neighbors. Greedy backward elimination with least-squares compensation might choose better units than my earlier selection, but at 3072 units and 10k tokens it would be expensive.

I'm now working out an efficient way to pick which of the 3072 units to drop: greedy backward elimination, where each candidate's removal cost comes from rank-1 updates of the inverse covariance rather than refitting from scratch. The step-by-step derivation of that error increase is what I'm setting up.

I'm reconsidering the OBS-style greedy selection: removing a unit costs its coefficient energy divided by the diagonal of the precision matrix. My earlier notes said that approach came out worse, and I'm wondering whether that was just overfitting to the finite token sample, since the ridge term should guard against it.

I'm settling on whether m=2048 with the new L0-L2 data and base-only L5 is worth preferring over m=1792, since the honest 1792 fit is only .981 and my earlier estimates put the test flip counts at roughly 9-12 for 2048.

I'm checking whether the edited model's logit shifts line up in one direction, since correlated flips toward a single class would cost more macro-F1 and could be fixed by a bias correction on the mean logit delta.

I'm asking whether the edited model's flips are systematically pushed away from the training labels on low-margin items, since the practice flips skew heavily toward wrong. If fine-tuning lives in low-variance units that pruning removes, the drift could make test flips more harmful than the -0.15 estimate. I want to check whether the logit changes lean toward or away from the true label beyond what chance would give.

I'm now splitting the low-margin items by whether the base model got them right, to see whether the edit changes the margin more negatively for correct base predictions than for wrong ones. If the drop is concentrated on the correct ones, that would suggest the edit is erasing fine-tuned memorization rather than just adding noise.



The honest m=1792 ceiling is 0.983 agreement, with 19 flips at 0.981 for the base-only fit, and pass 18's 0.985 was inflated by an in-sample fit. Next I'm checking whether the logit changes are systematic, such as a per-class shift an output bias could fix.



## tool_use: run_python

```json
{
 "code": "\nimport sys; sys.path.insert(0,str(WORK))\nimport numpy as np, json\nfrom lsbuild import gelu, layer_data\nfrom evalh import Lb, lab1, f1\ndef LN(x,g,b): mu=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-mu)/np.sqrt(v+1e-5)*g+b\npre='roberta.encoder.layer.5.'\nab,W1,b1,U,b2=layer_data(5,base)\ng2=base(pre+'output.LayerNorm.weight').astype(np.float64); bb=base(pre+'output.LayerNorm.bias').astype(np.float64)\nD=base('classifier.dense.weight').astype(np.float64); bd=base('classifier.dense.bias'); O=base('classifier.out_proj.weight'); bo=base('classifier.out_proj.bias')\nhead=lambda h: np.tanh(h@D.T+bd)@O.T+bo\nae=np.load('a5e_1792.npy').astype(np.float64); Ge=gelu(ae@W1.T+b1)\nL_full=head(LN(ae+Ge@U.T+b2,g2,bb))\npb=Lb.argmax(1); srt=np.sort(Lb,1); marg=srt[:,-1]-srt[:,-2]; i2=np.argsort(Lb,1)[:,-2]\nfor name,L in [('1792 fullL5',L_full),('pass12 m2048',np.load('pass_12/logits.npy')),('pass17',np.load('pass_17/logits.npy'))]:\n    d=(L-L.mean(1,keepdims=True))-(Lb-Lb.mean(1,keepdims=True))\n    dm=(L[np.arange(1000),pb]-L[np.arange(1000),i2])-marg\n    low=marg<1; r=pb==lab1\n    print(name,'mean centered logit shift per class',np.round(d.mean(0),3),' dm low&right %.3f low&wrong %.3f all %.3f'%(dm[low&r].mean(),dm[low&~r].mean(),dm.mean()))\n    pe=L.argmax(1); print('   pred counts base',np.bincount(pb,minlength=4),'edit',np.bincount(pe,minlength=4), ' flips to class',np.bincount(pe[pe!=pb],minlength=4),'from',np.bincount(pb[pe!=pb],minlength=4))\n"
}
```

