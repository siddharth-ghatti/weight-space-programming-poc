## reasoning summary

So with hs[9], after layer 8, plus the linear M1 carry, I get about 98.1%. That requires layers 0 through 8 plus the pool slot, so k=10, and at m=3072 that's 95.3M. The step from j=8 (97.7) to j=9 (98.1) is small but looks consistent despite the noise. I'm also wondering whether the quadratic kernel is really the best choice for the deeper features, and how a linear model would compare starting from the layers listed.

I'm weighing whether to keep all the BERT layers and just make the readout cheaper, or to trim the stack. The idea is to test dropping individual middle layers, since removing one or two would cut the parameter count a lot compared with aggressive FFN pruning, which seems to hurt quality badly.

I'm weighing whether to probe the FFN inputs for every layer in one pass by reading hidden states from an edited model. Since hs[L+1] is LN2 of the attention output plus the FFN output, I'd need to invert that relation to recover h_attn, which could let me prune FFN units far more accurately than per-layer probing, possibly saving 15-24M parameters.

I'm weighing an alternative to inverting the FFN residual map numerically: making the layer's FFN a no-op isn't workable, and reimplementing the forward pass iteratively is more like retraining than probing. A cleaner route might be the attention-only pass, but the h_attn it yields belongs to the attention-only network, not the real one.

I'm weighing whether a probe pass on one layer could recover its exact attention output, but that means ten separate passes, which is too many. Since the readout only cares about the final quality and the pruning is uniform across layers, I'm looking for a way to improve the cheap proxy instead, which currently ignores the attention output.

I'm weighing whether the top-768 units keep enough of the FFN signal, since they seem to retain only about half the importance mass in each layer. Deeper pruning looks likely to hurt too much, and at m=1536 the savings of around 23.6M would only hold if quality stays up, which seems doubtful. I'm now considering layer dropping as a way to test a different trade-off.

I'm thinking about dropping one or two middle layers to test whether that holds quality better than pruning FFN units, which seems to hurt too much. Since the attention-only and pruned-FFN tests both degraded the deep features, I'm going to try layer dropping to pick the layers that matter least.

I'm weighing a test that drops layers 3, 5, 7, and 9 instead of pruning FFN units, since that keeps a spread of depths that I can read out in one pass.

Comparing layer dropping against FFN pruning now: dropping one layer saves about 7M parameters and two layers about 14M, which lands near 80M, roughly the same savings as trimming FFN width to 2048 at the full depth.

I'm considering whether a linear map from shallow token states could approximate the deep features, but mean pooling is linear, so that wouldn't add the nonlinear contextual computation the deeper layers provide. The kernel on the shallow layer already reaches about 97.1%, which beats a linear readout on the deepest layer at 96.1%.

I'm considering whether the shallow readout could still gain from position information, since the layer-0 token states already carry position embeddings. A direction correlated with token index might let the heads weight early tokens differently from late ones, which could help with question-then-answer style sentences.

I'm weighing a plan given the limited passes left. The deep path, with layer dropping or FFN pruning, looks achievable near 80 to 88M at around 98% CV. The shallow path would save more parameters but sits around 97.2%, which likely fails, and a middle option at k=5 or 6 still looks likely to fall short.

I'm going to pursue the deep path and cut aggressively with the remaining passes, and maybe spend one or two on shallow pooling ideas. The readout matters here too, since the quadratic kernel gave about half a percent, so I want to confirm that a feature map I can actually implement with fewer Nyström units still reaches that level.

I'm noting that a GELU unit with a small scale behaves like a linear term plus a quadratic term in the dot product with a center, so the FFN units could reproduce the quadratic kernel's Nyström features. Using all the training points as centers would give exact kernel ridge in that span, but with m=3072 units I'd need fewer centers.

I'm checking whether the kernel readout survives the normalization change: the LN on the CLS state is applied to the raw vector, while the z-scoring can be folded into the FFN input weights, so the two may not match.

I'm checking that the centers and scales keep the GELU in its near-quadratic regime, so the Nyström units behave as the kernel features. Then the output layer's coefficients come straight from the kernel ridge fit, with the pooler and classifier passing the class scores through in the linear range.

I'm sketching the layer-dropping experiment now: test keeping layers 0, 1, 2, 4, 6, and 8 while removing 3, 5, and 7, which would bring the model to about 74M parameters.

I'm checking which prefixes of the layer pattern are worth measuring, since reading every depth in one pass lets me evaluate several candidate configs at once. Keeping layers 0 through 3 and then skipping every other one gives a 6-layer candidate near 74M, which is the one I want to test first.

I'm going to try a layer pattern that keeps layers 0 through 3 and then skips every other one, and I'll read the means at every depth for the 1024 ids in one pass. The FFN bias edits from the m=768 experiment are still in the saved model, so I need to restore the starting model before setting up the new structure.



Hidden state 9 with a linear carry of the layer-1 mean reaches about 98.1% CV; I'm removing the bias edits from the m=768 test and then measuring layer-skip patterns, starting with [0,1,2,3,5,7,9,10].



## tool_use: run_python

```json
{
 "code": "for e in list(edits()): drop_edit(e)\nprint(edits())\nprint(set_structure(layers=[0,1,2,3,5,7,9,10], ffn_keep=3072))\nwith open(WORK/'notes.md','a') as f:\n    f.write(\"\"\"- Exp A result: m=1 all layers -> deep means collapse (M9 krr 97.4->89.5 on first1024). Shallow ok.\n- pass4: token_hidden base, 64 ids (tok_ids.json) all layers -> ffn_imp.npz (proxy importance, E[gelu]).\n- Exp B (pass5): m=768 top-imp + bias folding, all 12 layers -> M9 krr 97.4->94.3. FFN pruning hurts deep features a lot.\n- KEY: shallow (M0/M1) quad-kernel ceiling ~97.1-97.3 CV (full set). Need M9 + lin M1 -> ~98.1. Pass bar likely ~start-2 errors -> want CV>=98.\n- quad kernel = K=C+0.5C^2 with C=cosine of z-scored, L2-normalized means; lam 1e-2. GELU taylor ~ lin+quad -> can implement as Nystrom in pool-slot FFN.\n- Exp C (pass6): layer drop pattern [0,1,2,3,5,7,9,10] full m, read means. (dropped all edits before)\n\"\"\")\n"
}
```

