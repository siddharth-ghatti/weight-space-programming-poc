## reasoning summary

I'm noting that the mean-comp error also falls with more training data, which I think means unit selection gets better and benefits from the extra data. The LS result at 3781 tokens is 0.164, about 8% better than mean-comp's 0.178, and I expect going to around 15k tokens could bring LS down to roughly 0.13.

I'm weighing whether a larger token budget would close the gap for the least-squares method, but I'm also questioning whether early-layer FFNs are inherently hard to prune at this width. Dropping a whole layer and widening the rest looks worse, since dropping layer 4 hurt agreement too much, so I'm likely stuck with roughly six layers. That means the achievable reduction depends on the width I keep, and I'm wondering whether a smarter unit-selection score that accounts for correlated contributions would help more than extra data.

I'm looking at greedy backward elimination as an alternative: at each step, drop the unit whose removal, with least-squares recompensation of the rest, raises error the least. That's the Optimal Brain Surgeon style score, ||w2_i||² / [G⁻¹]_ii, which is effectively the same LS compensation done greedily, and with limited data I'd need damping on the Gram matrix to keep it stable.

I'm weighing whether to avoid iterative fitting altogether by computing all the OBS scores s_i = ||w2_i||² / [G⁻¹]_ii in one shot, with G damped, ranking them once, dropping the lowest, and applying a single least-squares compensation. The scores should capture correlation between units, since a unit whose activation the others predict well has a large [G⁻¹]_ii and a low score. The catch is that the Gram matrix is estimated from limited data, so that correlation structure may be noisy.

I'm settling on the idea that more data is the real bottleneck, so I'm considering bigger probe batches: taking the 32 longest paragraphs, around 12,800 tokens per pass, and running two passes per early layer.

I'm estimating how the least-squares error might fall with about 18k tokens of data, perhaps to around 0.13 from 0.164, and I'm hoping the unit selection improves along with it. I'm also checking, from the real data, how the flips cluster at low margins before I settle on the noise requirement.

I'm wondering whether the test-margin distribution is knowable at all, and I'm leaning toward treating the practice examples the base model got wrong, the 75 hard cases, as a guide to where risk concentrates. Since the test thresholds are tight and a failing submission is the worst outcome, I'll aim for in-sample agreement of at least 99% and noise under roughly 0.1.

I'm estimating what the m=2048 model would do on the test set, using the low-margin practice bin as a proxy. In-sample accuracy drops from 92.5 to 91.7, which I read as mostly harmful flips, since the base model is right on nearly every training example. On the test set those flips should be closer to neutral. Scaling the in-sample flip rate by how much denser low-margin cases are in test versus practice gives roughly 4% flipped, about 16 examples, with an expected net effect near −0.5 points.

I'm wondering whether the flips on the test set really hurt as much as the in-sample numbers suggest, since the base model is nearly always right on training examples. Even so, with the estimated risk around 30 to 40 percent for m=2048, that seems too high, so I'm looking for lower-noise options. The most promising are exact removals: units that never activate, or that are constant, can be folded into the bias without changing any output.

I'm thinking about which units are nearly dead or effectively linear, since those could be removed or merged with little error. The variance ranking already captures the near-dead ones, and I'm estimating the error at larger removal counts for the early layers.

I'm considering whether some units are always linear or always dead, since those could be merged or removed with almost no error. I'd check the pre-activation distribution to count how many units fall into each regime before deciding whether that route is worth the effort. Otherwise, I'd accept some test risk at a moderate compression level.

I'm going to check the per-unit statistics next, to see how many units are effectively dead or always in the linear regime before deciding whether that route is worth pursuing.



More data reduces the mean-compensation error a bit, though gains look modest. Next I'll check each layer's units for dead, always-linear, and genuinely nonlinear ones before picking an approach.



## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'builder.py').read())\nfor Lr in range(6):\n    pre=f\"roberta.encoder.layer.{Lr}.\"\n    W1 = base(pre+\"intermediate.dense.weight\"); b1 = base(pre+\"intermediate.dense.bias\"); W2 = base(pre+\"output.dense.weight\")\n    A = ffn_data(Lr); Z = A@W1.T+b1\n    q01 = np.quantile(Z,0.005,axis=0); q99 = np.quantile(Z,0.995,axis=0); fpos = (Z>0).mean(0)\n    dead = (q99< -2.5).sum(); lin = (q01>2.5).sum()\n    H = gelu(Z); sc = H.var(0)*(W2**2).sum(0); sc_sorted = np.sort(sc)[::-1]; cs = np.cumsum(sc_sorted)/sc.sum()\n    print(f\"L{Lr}: dead {dead}, linear {lin}, frac-active median {np.median(fpos):.3f}, sparsity: units for 90/95/99% var-score {np.searchsorted(cs,.9)}, {np.searchsorted(cs,.95)}, {np.searchsorted(cs,.99)}\")"
}
```

