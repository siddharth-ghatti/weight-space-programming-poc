## reasoning summary

I'm comparing the m=1536 linear-units run against pass 3: its original head agreement is 0.922 and refit is 0.956, which isn't much better. Base practice accuracy sits at 0.925 with test at 0.775, so I can afford only about a 1-point drop to stay at 76.5%. Given that practice agreement near 0.92 probably implies a 2-4 point test drop, this setup looks risky.

In-sample practice accuracy fell from 0.925 to 0.861 under the original head, which is too large a loss. Next I'm weighing how far the model can shrink: the candidate configurations run from roughly 82M parameters with no reduction, down to about 68M at m=1536, with the one-layer-drop options landing in between. The test set is harder (77.5% against 92.5% on practice), so I'm wondering whether the pruning-induced prediction flips on test could go either way.

I'm reasoning that flipped test predictions are probably low-margin cases where the base model is only about 50% right, so each flip costs roughly 2-4 points of accuracy at 8% flips. That means I need the flip rate on test to stay under about 3%. The safest compression is probably a milder one: either m around 2048-2304 at k=6, or dropping a layer with a compensating adjustment. I've already seen that truncating to 5 layers with a refit head gives 0.955 agreement and dropping layer 3 with a diagonal LayerNorm compensation gives 0.944, so I'm weighing those against the parameter savings I'd get from each.

I'm weighing the cost of dropping a layer, which saves about 2.4M parameters plus 1537 per unit of m, so roughly 7.1M per layer at m=3072. Halving m across six layers would save about 14.2M. I'm also considering whether a moderate m like 2048 could keep agreement near perfect, and I want to check how much error each m introduces. With 14 passes left, I should reserve two or three for final verification.

I'm checking the sample count against the regression: about 4050 tokens per layer, which is too few to fit a 1536-dimensional least-squares refit without overfitting. So I'll lean on the moment-based approach with Gaussian approximations for the GELU statistics, and consider whether collecting activations across more passes would help.

I'm now weighing how the per-layer errors change with m, using the plan_layer output at m = 1536, 2048, and 2560. The aim is to see whether a moderate width keeps agreement with the base model close to perfect. Before settling on a compression level, I want to think about which measure actually predicts test accuracy, since practice agreement on in-sample items may not carry over.

I'm weighing whether a combined set of reductions could stay nearly lossless: m=2560 across six layers saves about 4.7M parameters, while dropping one layer saves about 7.1M. Both the five-layer truncation with a ridge refit (0.955 agreement, 0.82 on low-margin items) and the layer-3 drop with diagonal LayerNorm compensation (0.944) look too lossy, and the FFN pruning at m=1536 lands in the same range.

Checking whether the 76.5 threshold means only a 1-point drop, I'm weighing whether closed-form ridge refitting to the practice labels would help, though that data is in-sample. Test accuracy is 22.5% errors, so flips on test might partly correct those errors, and I'm wondering whether the net effect could be close to zero or even negative.

I'm weighing whether a more data-rich FFN compression is worth it, since each pass only gives about 32 ids of token-level activations, which is thin for fitting a 1536-dimensional refit. Given that limit, I'm reconsidering how much parameter saving is realistic against the agreement I can actually reach.

I'm looking at whether the last layer's FFN is the easiest to compress, since its output only matters at the CLS position and I have about 1000 CLS samples to fit against. The catch is that every kept layer has to share the same width m, so that advantage doesn't translate directly into a larger reduction.

I'm considering whether dropping the last layer would help, but the attention there is still significant, so that's ruled out. Layer dropping would need the next layer to absorb the missing residual update, and since post-LN blocks don't map cleanly to linear compensation, that approach looks hard to get right.

I'm now checking how much FFN output variance each layer's units account for, so I'll load the importance stats and the plan_layer errors at several widths to see which m keeps the error small.



Pass 11 at m=1536 is too lossy, agreeing with base only 0.922 overall and 0.72 on low-margin items, while the test allows just a 1-point drop. Next I'll check the per-layer FFN error across widths m to find the safe range.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nexec(open('lu.py').read())\nd=np.load('timp_L0.npz'); print({k:d[k].shape for k in d.files})\nfor l in range(6):\n    d=np.load(f'timp_L{l}.npz'); I=d['I']; NL=d['NL']\n    o=np.sort(I)[::-1]; c=np.cumsum(o)/o.sum()\n    row=[]\n    for m in (1024,1536,2048,2304,2560):\n        row.append(f'm{m}: plain {1-c[m-1]:.4f}')\n    print(l, ' | '.join(row))\n"
}
```

