## reasoning summary

Depth clearly helps accuracy, but nine layers at this width would land around 88M parameters, which is far over budget. So the saving has to come from narrowing the feedforward layers and maybe skipping some layers. The original's test accuracy may also be optimistic compared with cross-validation, so I need a fairer proxy to compare against. The 97.8% cross-validation estimate is the natural calibration point.

I'm checking whether the 97.8% cross-validation estimate is a fair anchor, since my linear-feature CV came in around 97.66%, which suggests the test score runs roughly half a point higher. I'll aim for CV of at least 97.8% rather than leaning on that offset, and I'm wondering whether narrower FFNs or skipped layers hold up. I also want to see whether mapping hidden states between layers with a ridge-fit linear map could stand in for a whole stack.

I'm checking whether a linear map can stand in for a stack of layers: a pair of GELU units with opposite signs can reproduce a linear term exactly, so a wide FFN could in principle implement one. But attention mixes tokens and LayerNorm sits in between, so the payoff is unclear, and I'd rather look at which features actually carry the extra two points over a bag of embeddings.

I'm weighing which FFN units to keep when narrowing the layers. Without real activations, I'd estimate their importance from the output weight norms scaled by a typical activation size, using the previous layer's hidden states as a stand-in for the input distribution. The plan is to test this on a prefix of about the first nine layers and see how much the features degrade.

I'm weighing whether the activation statistics I'd need for pruning can come from token-level hidden states at each layer, using the residual stream as a stand-in for the post-attention LayerNorm input. Mean-pooled states look too smooth to be a good proxy, so token states from one pass over 64 sentences would be the better sample. I'm still unsure whether approximating the FFN pre-activations this way amounts to reimplementing the forward pass rather than just estimating a proxy.

I'm weighing how to score which FFN units to keep when narrowing the layers. A weight-based or activation-based importance estimate could work, since the classifier gets refit anyway and what matters is preserving information.

I'm comparing layer-count and FFN-width combinations against the 24.43M-parameter base plus a per-layer cost. Several options land near 50M or below, such as 8 layers at 512 FFN width or 4 layers at 768, and a few sit right at 52.8M.

I'm going through the compute budget for a couple of smaller configurations, looking at how dropping layers versus narrowing the feedforward units trades off against the parameter cap. Dropping layers saves the most when the units per layer are already small, so I'm planning a few screening runs before committing to a full-data evaluation.

I'm weighing a quick screening run: all eleven layers at a narrower feedforward width of 1024, read at every layer, to see how sensitive each depth is to FFN narrowing. Then I'd compare that against a skipped-layer stack, such as keeping layers 0, 1, 2, 4, 6, 8, and 10 at width 3072.

I'm weighing how to choose which FFN units to drop, and a bias correction for their mean contribution is the simplest option. A better version would fit the removed units' output with a least-squares combination of the kept units' activations, adjusting the kept weights to compensate.

I'm weighing whether the hidden states from the previous layer can stand in for the FFN input, since the real input would need attention to compute. Since the residual stream dominates, that proxy might be good enough to rank units, though not to reconstruct their outputs exactly.

I'm considering whether I can expose the attention-LayerNorm output directly by temporarily setting its gain to one and its bias to zero, so the layer's output reveals the normalized FFN input.

Editing each layer's output to read out its intermediate activations would mean a separate pass per layer, since changing one layer changes everything downstream. That's expensive, so I'm weighing whether reading the FFN inputs at all is worth it, or whether the residual-stream proxy is good enough for ranking units.

I'm wondering whether a shallower design can work, since layer cost dominates the parameter budget. Two layers with a nonlinear readout on the first two layers' means gave about 97.4% in one unnormalized run, which is close to the six-layer prefix. Adding attention-pooled CLS features from layer two only brought it to 97.1%, so that isn't clearly helping.

I'm considering whether a 12-layer model with the FFN fully removed would hold up, and whether keeping only four layers at full width gives the same 52.8M budget. I'm also wondering whether the errors point to a semantic pattern, like the question about disclosing a risky hobby being mispredicted as an exclusion.

I'm thinking about whether to skip layers or narrow the FFN width, and I want to test both. I'll try an experiment that keeps only every other layer (0, 2, 4, 6, 8, 10) at full width, and another that keeps all eleven layers at width 768, reading the mean and CLS features at each edited layer.

I'm checking whether dropping the reserved-dimension edits from layers 2 through 10 changes anything in the experiments. Those edits only cut reads and writes on the reserved dimensions, so they shouldn't leak into the other features, and I'll mask those dimensions out of the feature set regardless. Dropping layer 1 would let the reserved dimensions carry ordinary BERT content into layer 2, which is a minor concern. I'm still deciding whether to restore the original BERT weights before running them.

Next I'm settling the layer design: the kept layers form a set S, and the last one plays the pooling role the original layer 11 had. The carry heads from the start model can't be restored, but I can reuse the reserved channel and refit them.

I'm thinking about keeping layer 11 as the final layer, since it already does the pooling. Its FFN output has been zeroed, so only its bias contributes, which means its unit count doesn't matter and I'd still need to keep the uniform width. The pooler would then be refit with ridge on the final CLS state and the decoded reserved channel.

I'm weighing whether the last real layer could take over the pooling role, with some heads attending normally for CLS and others averaging uniformly over tokens, instead of spending a whole separate layer on pooling.

I'm now asking whether the nonlinear head could work better if the network itself computed nonlinear features, rather than carrying only the eight linear ridge scores through the reserved dimensions.

Next I'll run some experiments to see how much FFN narrowing and layer skipping cost, since those decide the parameter budget, and only then settle the head design. Before that, I need per-unit activation scores, so I'm pulling token-level hidden states for 64 sentences across all 13 layers and using each layer's input as a rough stand-in for the FFN input.

I'm weighing a simpler proxy for ranking FFN units, using the weight norms of the input and output projections instead of measured activations, though activation-based scores usually work better. Before settling on a scoring method, I'll pull token-level hidden states for a stratified sample of 64 sentences, eight per class.



Accuracy only reaches about 97.9% around layer 9 or deeper, so savings must come from narrowing the FFNs and possibly skipping layers. I'm pulling token-level hidden states at every layer for 64 stratified sentences to pick which FFN units to keep.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\ny=labels(); rng=np.random.RandomState(1)\nsel=np.concatenate([rng.choice(np.where(y==c)[0],8,replace=False) for c in range(8)])\nids=[f'p{i:04d}' for i in sorted(sel)]; print(json.dumps(ids))\njson.dump(ids,open('tok_ids.json','w'))"
}
```

