## reasoning summary

The pruned model at m=1536 with weight-only importance looks essentially broken: its logits are nearly constant across inputs and it always predicts the same strategy, whereas the base model gives a confident prediction on p0000. So choosing units by weights alone at that size seems to destroy the model.

I'm now weighing the options after this result: the weight-only pruning at 50% clearly breaks the model, so I shouldn't submit this structure or prune FFN blindly. The base model stays as the safe fallback, and I could consider only a much milder reduction.

I'm working out whether a rigorous weight-only bound can flag units that are effectively dead: layer-normalized inputs have fixed norm, so Cauchy–Schwarz caps each unit's pre-activation, and units capped well below zero contribute almost nothing through GELU. I'm checking how many units that criterion would actually catch, since the bound is loose.

I'm checking whether the FFN importance estimate itself has a bug: the mean uses the attention output LayerNorm bias, which looks right, and the spread is the root of the sum of squared weights times gamma squared. Those both seem fine, so I'm weighing whether the energy term ignores the actual input distribution in a way that matters.

I suspect the Gaussian assumption is what's breaking the importance estimate: the LayerNorm output has a few outlier dimensions that carry most of the energy, so units reading them stay consistently active, and my estimate undercounts them. That would explain why pruning them destroyed the model.

I'm wondering whether I can sharpen the importance estimate using the embedding layer, since its weights are available to me. Computing the layer-0 input would let me see the real activation statistics rather than assuming Gaussian inputs. But going further through attention and the rest of the stack would mean reimplementing the forward pass, which the rules rule out.

I'm deciding against any forward-pass shortcut here, since that would mean approximating the attention layers by hand, which I've ruled out. So I can't verify a pruned version, and the one result I have says weight-only pruning broke the model. The safe choice is to submit the unchanged base model: 82,301,956 parameters, which matches the starting accuracy of 77.5% and 0.7096 by construction, with no reduction.

I'm considering whether any reduction could be provably lossless from the weights alone, like FFN units with exactly zero or near-zero output columns. I'll check the weight norms to see if there are any such dead units before deciding anything else.

I'm checking whether any units have negligibly small output weights, since those could be dropped with little effect. For each unit I'd bound its contribution using the Cauchy–Schwarz estimate on its pre-activation and the norm of its output column, then compare the total against a typical hidden-state norm.

I'm going to check whether any FFN units are rigorously negligible: bound each unit's output by its GELU range under the Cauchy–Schwarz pre-activation estimate, times its output column norm. Only units whose summed bound stays tiny relative to the residual norm could be dropped, and since the bound is loose, I expect the savings to be small, maybe around 3%.

I'm checking whether a rigorous bound on each unit's output can justify dropping the ones whose contribution stays negligible. The LayerNorm normalization caps the input norm, so each pre-activation is bounded by its gamma-weighted weight norm, and I'd use that to bound the GELU output.

I'm thinking about bounding each pruned unit's output by its worst-case GELU value over the Cauchy–Schwarz interval, then summing those bounds through the output columns. The idea is to keep adding the lowest-bound units until the total perturbation stays under a small fraction of the residual norm, though I still don't know what that norm is.

I'm estimating how much a provably safe reduction could save: the LayerNorm input has bounded norm, so each FFN unit's pre-activation sits in a known interval. I'll check whether any units with tiny worst-case output could be dropped, and I expect that to only remove a modest slice of the FFN, perhaps 10–30% if some units are effectively dead.

I'm settling on a plan to pick a common pruning width: since every layer has to shrink by the same amount, the layer with the fewest provably safe units sets the limit, and each dropped unit would save about 1,537 parameters per layer across the six layers.

I'm checking whether a rigorous L2 bound on the pruned units' contribution keeps the logit change small, around a percent of the residual scale. That would only flip predictions near the decision boundary, while still cutting roughly 3–6% of the parameters.

I'm going to keep the reduction rigorous rather than guess: I'll bound each unit's contribution using the layer-norm constraint on its input and drop only the units whose total worst-case effect stays small. First, though, I'm resetting the structure to the full model so that a submission can't be left in the broken pruned state.



The 10-paragraph check showed my weight-only FFN ranking broke the 6-layer model, so I'm resetting structure.json to the full structure and won't submit the pruned version. With no more forward passes, I'll look for FFN units that are provably near-dead using a worst-case bound on the weights.



## tool_use: run_python

```json
{
 "code": "\nimport numpy as np\nclear_structure()\nprint(param_count()['total'], structure()['m'], structure()['layers'])\nopen('notes.md','a').write(\"\"\"\n- FINDING: edited 6L m=1536 (gaussian-energy pruning) is BROKEN: logits ~constant [-0.4,0,0.9,-0.4], always 'strategy' on 10 items; base p0000 was [-1.46,-1.39,-0.27,3.78]. DO NOT USE.\n- structure reset to full (clear_structure). Fallback = unpruned base (passes by construction).\n- Next idea: only prune FFN units that are PROVABLY ~dead from weights alone (Cauchy-Schwarz bound using LN: ||z||<=sqrt(768)).\n\"\"\")\nfrom scipy.special import erf\ndef gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))\ndef gelu_absmax_interval(lo, hi):\n    # max |gelu| on [lo,hi]; gelu has min ~ -0.17 at x~-0.75; monotone for x>-0.75 ; for x<-0.75 |gelu| decreasing in |x|... compute on grid\n    xs = np.linspace(0,1,201)[None,:]\n    g = np.abs(gelu(lo[:,None] + (hi-lo)[:,None]*xs))\n    return g.max(1)\nres = {}\nfor l in range(6):\n    p = f'roberta.encoder.layer.{l}.'\n    Win = base(p+'intermediate.dense.weight'); b = base(p+'intermediate.dense.bias')\n    Wout = base(p+'output.dense.weight')\n    g = base(p+'attention.output.LayerNorm.weight'); be = base(p+'attention.output.LayerNorm.bias')\n    mu = Win@be + b\n    r = np.sqrt(768.0)*np.linalg.norm(Win*g[None,:],axis=1)\n    amax = gelu_absmax_interval(mu-r, mu+r)\n    bound = amax*np.linalg.norm(Wout,axis=0)\n    res[l]=bound\n    xs = np.sort(bound)\n    cs = np.cumsum(xs)\n    xnorm = np.sqrt((g**2).sum()+(be**2).sum())\n    print(l, 'ref |x|~%.1f'%xnorm, 'n(bound<1e-3)=',(bound<1e-3).sum(), 'n(<1e-2)=',(bound<1e-2).sum(), 'cumsum at 100/300/600: ', cs[[99,299,599]].round(3), 'min wout norm', np.linalg.norm(Wout,axis=0).min().round(4))\nnp.save('rigorous_bound.npy', np.stack([res[l] for l in range(6)]))\n"
}
```

