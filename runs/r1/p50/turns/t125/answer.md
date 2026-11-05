## reasoning summary

I'm checking the systematic part of the error, which is small and only about 7% of the variance. The positive correlation with the edited logits looks like regression to the mean rather than anything meaningful. So the remaining noise seems essentially random relative to the logits.

Now I'm weighing whether a more fundamental gain is available at the same m, and the first candidate is token weighting by CLS attention. Early-layer errors reach every token downstream, so weighting only pays off at layer 4, where it might recover maybe 5% of the total noise, which is marginal. A second idea is to project layer-4 output errors through the layer-5 value and output maps onto the few logit directions that actually matter.

I'm now considering whether the downstream metric could make layer-4 pruning far more effective: only a small, roughly 20-dimensional slice of layer-4 output errors actually reaches the logits through the layer-5 CLS attention. Selecting units to preserve that slice might beat the 20% gain I got from the post-LN weighting. Still, layer 4 is only about a fifth of the total noise, while layers 0 and 1 carry about half and propagate everywhere.

I'm settling on the m1536 model and still need to decide between v1 and v3. Before that, the choice of m comes down to how likely a failure is at each size, so I want a more careful risk estimate that uses the test flip distribution and the direction of the flips.

I'm estimating the test flip counts for the m1536 model at roughly 8 to 11, which puts the failure risk around 10 to 18%. That feels high, so I'm comparing it against m2048, which should cut flips to about 6 to 8 and risk to roughly 5 to 10%. Whether the smaller model is worth that risk depends on how the score trades off passing against parameter count, and I don't know that yet.

I'm thinking about whether the L5 feed-forward layer could be repurposed, since it only really affects the CLS token and so its unit budget might be better spent on a correction that maps the edited CLS attention output toward the base model's output.

I'm weighing the m1536 choice against the failure risk, and I'm working through how many test flips that size would produce, using the margin noise seen in practice.

I'm weighing whether to stop here with m1536 or spend a pass on a downstream-metric selection at layer 4, which might cut noise by 5 to 10 percent. Another option is to try dropping most of layer 5's FFN, since it only shapes the CLS token.

I'm weighing the parameter savings from shrinking m to 1280, which would cut about 2.4M parameters, roughly 2.9%. Given how flat the risk-versus-size tradeoff looks, m1024 through m1536 seem roughly equal in expected value, so I lean toward m1536 for lower failure risk. Going down to m1024 would save about 7% but raise the failure chance from around 13% to 25%.

Weighing the parameter savings against failure risk, m1280 would cut about 20% of total parameters, but I'd need to look at the noise in that range. I'm thinking about whether a smarter compensation at fixed m could lower the noise enough to shift the risk tradeoff, and where that noise actually comes from.

I'm considering whether the FFN input could serve as a regressor for the dropped units, since their output would then be approximated linearly in that input. The catch is that the FFN has no linear path from its input, and the LayerNorm before the attention output makes it awkward to fold any such term back into the weights.

I'm looking at whether the compensation for dropped units can be improved by choosing which units to keep, since units that are nearly linear or near-constant can be absorbed cheaply. The greedy selection already accounts for this through the conditional covariance of the dropped outputs. The bigger issue seems to be that with roughly 1536 regressors and only about 15k tokens, ridge overfitting is substantial, with train error around .23 against test error around .307 at m1024 in layer 1.

I'm wondering whether the regressor count could be cut by restricting compensation to a low-rank map or to the kept units most correlated with the dropped ones, though ridge already handles much of that. Getting more fit data by probing base activations for all the practice paragraphs doesn't look workable, since token outputs cap at 32 ids per pass and mean outputs lose the per-token detail I need. I'm also considering whether layer-level probes could expose the FFN input directly, which would give more tokens per pass.

I'm wondering whether the probes could return enough per-token second-moment information to fit the compensation regression. Mean pooling only gives a linear statistic per sequence, and the gelu units make the needed products nonlinear. Packing several layers into one pass would still leave far fewer numbers than the roughly 1.2M entries of X^T X.

I'm thinking about whether to gather more probe data for layers 0 and 1, since they carry about half the noise, and I'm counting the tokens I already have against the unused probe ids.

I'm working out how many probe tokens I'd get from more passes: about 6k per pass on the next-longest sequences, so roughly 18k extra for each of layers 0 and 1. Then I'm checking the learning curve for layer 1, where error drops only slightly from 12.6k to 16.9k tokens, to estimate how much that extra data would actually help.

I'm estimating the remaining gain from more probe data at about 2% of total noise, which isn't worth the extra passes. The error seems to plateau near 0.093 for layer 1 at m1536, so the rest is bias from the linear compensation rather than a lack of data. I'm now considering whether a nonlinear compensation, such as adjusting the kept units' input weights, could reduce that bias.

I'm considering whether merging similar units could reduce the bias, since two units with close input weights could be replaced by one representative unit with summed output weights. Still, I think the greedy selection plus least-squares compensation already captures most of what merging would offer.

I'm considering whether to reconstruct new units in closed form, rather than keeping original ones, by overwriting the kept rows of the first projection, but I don't see a clean closed-form way to pick them. Random features with least squares would likely do worse than the units already kept, so I'm treating the per-layer approximation as near its limit and turning to how to pick m.

I'm weighing whether to spend a few more probe passes checking how robust m1536 is against a larger m1792, which would cost about 14% of the parameters. I'm also trying to estimate how the edited model's test accuracy shifts from the base, since the practice-data flips may not reflect what happens on test-like inputs.

I'm settling on the view that a smaller parameter count with lower noise isn't available: dropping a whole layer to five layers at m2048 would cost more error than halving the FFN width does, and the wider FFN adds noise of its own. So I'm estimating the failure risk at roughly 10-15% for m1536, versus about 6-9% at m2048.

I'm comparing the v1 and v3 candidates for the final choice: v3 has a lower holdout noise variance, while its practice agreement is slightly worse, though the difference looks statistically insignificant. To look at which one is more robust, I'm planning to count how many items would flip if their base margins were halved.

I'm checking how the v1 and v3 candidates differ on items with small margins, since an ensemble of their edits isn't possible with different keep sets. I'll run a stress test that halves each base margin and count how many items would flip under that, to compare robustness.



The logit error is essentially random noise, with a linear correction barely moving rms (0.170 to 0.164), so a head fix won't help. Next I'll stress-test v1, v2 and v3 with smaller base margins and count the items that would flip.



## tool_use: run_python

```json
{
 "code": "bl=np.load('base_logits.npy'); y=np.load('y.npy')\ndef stress(lg, ks=(1,1.5,2,3)):\n    pb=bl.argmax(1); out=[]\n    for k in ks:\n        # shrink base logits by k around their mean, add the same noise d\n        d=lg-bl; b2=bl/k; e=b2+d\n        out.append(int((e.argmax(1)!=pb).sum()))\n    return out\nfor f in ['lg_pass25.npy','lg_pass26.npy','lg_pass29.npy','lg_m1536_v2.npy']:\n    lg=np.load(f); print(f, stress(lg), 'rms',np.sqrt(((lg-bl)**2).mean()).round(4))\n# also earlier m2048 pass12?\nimport glob\nfor p in ['pass_12','pass_10','pass_11']:\n    try:\n        pr=json.load(open(p+'/predictions.json')); idx={r['id']:i for i,r in enumerate(pr)}\n        ids=json.load(open('all_ids.json'))\n        lg=np.array([pr[idx[i]]['logits_edited'] for i in ids])\n        print(p, stress(lg), np.sqrt(((lg-bl)**2).mean()).round(4))\n    except Exception as e: print(p,e)\n"
}
```

