## reasoning summary

I'm working out how the reserved dimensions shift each LayerNorm: the shared offset of about 3.4 at L9 nudges the mean by roughly 0.004 and the variance by about 0.002. The effect on other dimensions is small, around 0.1 to 0.2 percent, and the label-specific part is smaller still.

I'm checking whether a tiny common shift across the 760 dims could be amplified by per-dimension z-scoring, and it only works out to about 0.0015 z units per dim. Since that shift mostly reflects collapsed class scores, it's probably only weakly informative. The editor's decoder subtracting the mean across 8 dims suggests the shared offset is large while the class-specific part is small. I'm also fairly convinced the jump at L9 is real rather than gradual.

I'm settling on a total of 95,312,648 parameters with k=10 and m=3072, since there's no further saving available from pruning at those settings. I'm also weighing whether a data-driven pruning criterion could beat the Gaussian proxy, perhaps by using per-unit statistics from the actual layer inputs.

I'm weighing whether a better pruning method using per-unit statistics from the layer inputs could justify keeping m at 2816 instead of 3072, which would save about 4M parameters. The payoff is real but the extra passes are tight against my budget, so the risk is worth weighing.

I'm weighing whether to spend a verification pass on the m=3072 head, since the plan needs one read of logits for 1024 ids and possibly a second for the remaining 472. I'm also wondering whether combining the L1 mean-only result (97.93) with the L9 features would help, though I haven't settled that yet.

I'm settling on m=3072 and staying put: pruning to 2816 would save about 4M parameters, but a better criterion needs extra passes I can't spare, and the margin matters more than the size gain.

I'm weighing a looser plan that drops the pruning edits and returns layers 0 through 8 to their base form, keeping all 3072 units. From there I'd compute the final CLS features for all 1496 items, then fit the head, either as a GELU kernel in layer 11 or through the pooler.

I'm now thinking through how the final head would sit in layer 11: the pooling output feeds the FFN, and I'd need to set the LN2 gamma on the reserved dimensions so the kernel scores pass through to the pooler.

Now I'm setting the layer-11 FFN so each unit computes a scaled dot product between the centered, standardized pooled features and a kernel center, with weights and biases folded in from the center's values and the training means and variances.

I'm recomputing the exact FFN input x1 in numpy from the base reads, normalizing over all 768 dims with the carried CLS values in the reserved slots, rather than using my non-reserved-only features. Then I'll fit the kernel on those features directly.

Now I'm working out how the LN2 statistics come out once the reserved dimensions carry the kernel scores, since the non-reserved mean is tiny and the reserved slots dominate the normalization.

I'm checking that the final head's logits stay clean: the pooled outputs for the eight reserved dimensions reduce to tanh of a scaled difference from their mean, so the argmax still tracks the kernel scores. The remaining concern is float32 precision, since the ridge coefficients could be large enough that the summed scores grow sizable.

I'm checking whether float32 precision holds up if the ridge coefficients get large, by comparing the float32 scores against float64. I'm also noting that the LN2 scale doesn't affect the argmax, since the decode step subtracts the shared mean and the tanh just needs to stay out of saturation, so I'll pick a scale that keeps the values within about ±2.

I'm now comparing a few ways to fit the head on the exact layer-11 inputs: kernel ridge with an RBF kernel, plus a GELU variant using all the centers, checked with both leave-one-out and 5-fold splits. The GELU version might also get two units per center, which would still fit within the 3072-unit budget.

Next I'll set up the ridge fit on all 1496 items as the deployed head, mirroring the cross-validation with the training fold as centers. I'll also check that the hyperparameters sit in a flat region, and confirm the base read ordering matches the labels before I compute the layer-11 inputs.



Layers 0 through 8 stay unpruned, with layer 11 as the head, since pruning at 2560 costs 0.5–0.8 and 2816 leaves too thin a margin. I'll note this decision first.



## tool_use: run_python

```json
{
 "code": "s=open('notes.md').read()\ns+='''\n## Exp3 result (pass_5, m=2560): idx9 subset LOO 97.27 vs base 97.75; 5-fold x5 96.3 vs 97.1 -> REJECT pruning (loss 0.5-0.8).\n- Importance flat (bottom512 = 8-10% energy, bottom256 = 3.5-4.5%) -> m=2816 likely ~-0.3. Not worth it.\n- Verified: pooling layer output == LN2(LN1(C9 + M9@O.T)) exactly (rel 1e-7) -> for UNPRUNED 0..8 compute final feats from base reads, no pass needed.\n- Full LOO depth profile (mix): L1..L8 ~97.3-97.9, L9 98.66, L10 98.46, L11 98.3. Leak via LN stats negligible.\n- GELU features at all centers (g=1,b=0,lam=.03) 5fold 98.17 ~= RBF KRR 98.2. Idea: pooling layer FFN (3072 units) = kernel head, writes scores to RES dims, pooler rows0-7 decode (mean-sub) -> argmax exact.\n## DECISION: structure [0..8,11], m=3072 (95,312,648 params). Drop pruning bias edits on 0..8.\n'''\nopen('notes.md','w').write(s)\nfor L in range(9):\n    drop_edit(f'bert.encoder.layer.{L}.output.dense.bias')\nprint(set_structure(layers=[0,1,2,3,4,5,6,7,8,11], ffn_keep=3072))\nprint(edits())\n"
}
```

