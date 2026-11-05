## reasoning summary

The gains are modest: head agreement only moves from 91.7% to 92.2%, and after refitting it reaches 95.6% versus 93%. The CLS error at layer 1 is still about 5.4 times the cross-sample spread, so CLS is clearly the position that is badly perturbed early on. Since CLS barely varies across inputs at layers 1 to 3, a systematic error there is the likely cause, and I'm checking whether its FFN input looks very different from that of ordinary tokens.

I'm working out why the CLS error persists: its FFN input is nearly constant across inputs, so pruning produces a constant error vector at CLS that the usual token-average compensation can't correct. I'm checking whether the units active for CLS are the ones being pruned, and whether choosing units that matter for CLS would reduce that coherent error.

I'm now weighing a mixture view: CLS tokens as their own subpopulation alongside regular tokens. For each unit I'd score how far its CLS mean sits from the token-wide mean, since the CLS errors from pruned units add up coherently in one constant vector.

I'm wondering whether the CLS error actually matters as much as the other positions, since special tokens like </s> and CLS act as attention sinks in the middle layers and their errors spread into every token's attention values. I'm also considering whether a linear-unit trick could correct CLS specifically, since linear units implement a shift relative to a mean.

I'm considering whether a linear unit could correct the CLS error specifically: a direction that is near zero for regular tokens but large for CLS would let a single unit push back on that constant error. Alternatively, a nonlinear unit that fires only for CLS, with a negative bias on a CLS-specific direction, could output the needed correction vector.

I'm weighing a CLS-specific detector unit, one that fires only for the CLS direction and outputs the needed correction, which could fix the constant CLS error with one or two units per layer. I'm also wondering whether the </s> token and other frequently attended tokens need the same treatment, so I'll check how large their errors are.

I'm going to check the per-position errors at each layer, comparing the edited model's token states against the base model's for the 32 probe ids, to see whether CLS is the only position that needs fixing or whether </s> and punctuation tokens have large errors too.

I'm checking whether a CLS-specific correction is needed, since the constant CLS error from pruning isn't fixed by the token-average compensation. The linear units also extrapolate poorly to CLS because it's an outlier far from the token mean. I'm considering a CLS-detector unit that outputs the needed correction, with the error estimated from the true versus pruned FFN output at CLS.

I'm weighing whether to compute the CLS FFN activations in closed form from the Gaussian statistics at CLS, which would be essentially the same as evaluating at the mean. Another option is to skip that and measure the CLS error directly from the forward pass, though those post-LN states already include upstream errors from earlier layers.

I'm weighing whether the CLS error can be recovered directly from the layer-0 outputs, since the attention in that layer is unchanged and the embeddings match. That gets messy once LayerNorm and the residual are involved, and errors compound at deeper layers anyway, so I'm leaning toward a simpler CLS-specific correction.

I'm now planning to fit the CLS statistics separately, using the 32 probe samples at position 0 for each layer 0 to 4, since the CLS spread is tiny. The layer 5 case already has 1000 CLS samples, so that part is set.

I'm settling on a simpler fix: add one or two CLS-detector units per layer whose output exactly cancels the difference between the full and pruned FFN outputs averaged over CLS samples. That should remove the constant CLS error without relying on the linear units to extrapolate to this outlier token.

I'm writing the CLS error as the kept units' output plus the linear units' output at CLS, minus the full FFN output, all computed from the CLS Gaussian moments. I'm also planning a detector unit with a direction along the LDA-style vector between the CLS and token means, so it fires only for CLS.

I'm sketching a detector unit for the CLS direction: project all tokens onto a direction like the mean difference or LDA vector, then pick a threshold so only CLS fires strongly. The output column would be the needed CLS error vector scaled by the gelu response at the detector's active point.

I'm checking whether the </s> token is also an outlier like CLS, since it's another attention sink whose error could spread through attention. I'll measure how far CLS and </s> sit from the token distribution, and compare the CLS error against its spread, using the existing data first.



Fidelity reached 92.2% and 95.6% after refit, but CLS errors at layers 1–3 run 5–7 times their cross-input spread, so a dedicated detector unit could correct them. I'm checking how separable CLS and `</s>` are from ordinary tokens in each layer's FFN input.



## tool_use: run_python

```json
{
 "code": "import json\ntoks=json.load(open('pass_2/tokens.json')); mask=np.load('pass_2/attention_mask.npy').astype(bool)\nprint(type(toks), len(toks), toks[0][:5], toks[0][-3:] if isinstance(toks[0],list) else '')\n# build position labels over masked tokens\npos=[];kind=[]\nfor i in range(mask.shape[0]):\n    T=mask[i].sum()\n    for t in range(T):\n        tk=toks[i][t]\n        kind.append(0 if t==0 else (1 if tk=='</s>' else 2))\nkind=np.array(kind); print(np.bincount(kind))\nnp.save('tok_kind.npy',kind)\nfor l in range(5):\n    a=np.load(f'true_a_L{l}.npy').astype(np.float64)\n    mt=a[kind==2].mean(0); \n    for k,name in [(0,'CLS'),(1,'</s>')]:\n        mk=a[kind==k].mean(0)\n        d=mk-mt\n        p=a@d; thr_lo=p[kind==2].max(); thr_hi=p[kind==k].min()\n        print(f'L{l} {name}: ||mean-tokmean|| {np.linalg.norm(d):.1f} vs tok spread {np.sqrt(np.trace(np.cov(a[kind==2].T))):.1f}; proj sep: max other {thr_lo:.1f}, min {name} {thr_hi:.1f}, within-{name} spread {np.sqrt(np.trace(np.cov(a[kind==k].T))):.2f}')"
}
```

