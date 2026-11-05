## reasoning summary

I'm checking notes.md in case this is a restart, then looking over the working folder. My plan is to aim for a compact model: drop some layers and narrow the feed-forward network, using closed-form compensation since there's no training. Pruning FFN units would need activations, which I can't get without running the model, so that part is still an open question.

I'm weighing whether to fit the new classifier head in closed form on the cls hidden states from the edited model. Ridge regression fits that, while logistic regression would need iterative training, so I'm leaning toward ridge. With only 1000 practice paragraphs against 768 dimensions, I'd want strong regularization, and regressing onto the base model's pre-tanh output seems a safer target than the labels.

I'm weighing a closed-form head refit: ridge regression on the edited model's last-layer CLS states, targeting the base model's pre-tanh output, since the dropped layers change the hidden distribution. A linear map from the edited hidden states to the base hidden states would fold into classifier.dense, though fitting per-token maps is limited by how many token vectors I can pull per pass.

Working out which FFN units to prune is harder than it looked, since their importance normally depends on activations I can't read. I could reconstruct the attention sublayer from token hidden states, but that would amount to reimplementing the forward pass, which is off-limits, so I'm inclined to stick with weight-based importance.

I'm wondering whether computing FFN activations from the returned hidden states would amount to running part of the model, which I'd rather avoid. Using the layer input as a stand-in for the FFN input is still evaluating GELU over the first FFN projection in numpy, so I'm leaning toward keeping FFN importance purely weight-based.

I'm treating the per-unit importance as a closed-form estimate from activation statistics rather than from running the model. With an input mean and covariance, each FFN unit's pre-activation mean and variance follow directly from its weights, which gives an approximate firing rate and output magnitude without reimplementing the forward pass.

Using the layer input as a proxy for the FFN input is reasonable here, since post-LN hidden states change slowly between layers. Token-level hidden states for around 32 ids would give roughly 3000 vectors for estimating covariance, while the masked mean per layer only gives the first moment.

I'm weighing whether a bias-compensation step, folding each pruned unit's mean output into the output bias, is enough, or whether a least-squares reconstruction from the kept units would be worth the extra cost. I'm also checking what a realistic size target looks like, starting from the full per-layer feed-forward parameter count, and whether a width of zero would even be allowed.

I'm sizing the candidates against the parameter budget: each attention block is about 2.37M parameters and each FFN unit about 1,537, so three layers at 1024 units comes to roughly 51.6M and two layers at 512 units to about 46.1M. Keeping zero layers would leave a constant classifier input, so that option is useless, and one layer sits at a floor of about 42.1M.

I'm checking whether the tight accuracy margin holds up, since the test set is roughly 400 examples and one point is only about four of them. Dropping layers without training usually costs a lot, though a closed-form head refit on the 1000 practice paragraphs might recover some of it, so I'm weighing whether a ridge or LDA head on the CLS representation could get close to the original 77.5%.

I'm considering whether zeroing the last layer's query weights would make CLS attend uniformly, so its output becomes a mean of value vectors and effectively mean-pools the sequence for the head. Since only the CLS position matters in that final layer, changing its attention there seems safe.

Since the attention scores are zero for every key, the softmax comes out uniform over the real tokens, so CLS in that final layer ends up as a mean of the value vectors. Zeroing the key weights while keeping the key bias would also give uniform attention. Either way the head sees a mean-pooled signal, and that reading still runs through LayerNorm and the FFN, so I'd fit the head on the edited model's actual final-layer CLS states rather than on a closed-form guess.

I'm weighing a layer-dropping experiment with a refit head, since the top fine-tuned layers hold the task signal and middle-layer removal may cost less. My plan is to run a few passes testing which layers to keep.

I'm planning a first pass that reads the base model over all 1000 practice ids and saves CLS and mean hidden states for every layer. A linear probe on each layer's states should show how much accuracy truncation can keep, since layers before the cut are unchanged.

Now I'm checking whether a probe on the practice data's CLS features can estimate how well truncated layers would generalize, since the base model fits the practice set much more tightly than it does the test set.

I'm comparing a probe on the base model's layer-6 CLS features against its 77.5% test accuracy, to see whether the probe's cross-validated score on practice data is inflated. Base predictions agree with the labels at about that rate, so if the probe tracks base predictions closely, the test score should land near 77.5% with a few points of spread either way.

I'm planning to check the base model's accuracy on the practice set first, since its logits there will show how closely the labels match before I choose a distillation target. For the candidates, I'm weighing truncation, skipping a layer, narrowing the FFN by importance, and refitting a ridge head on the final CLS states, with the practice-set accuracy computed from those saved features.

I'm checking the head's structure, which is a dense layer with tanh followed by an output projection. Since the head itself is simple, I can treat a linear probe as just a set of weights to fit on the activations, and I'd still verify the final numbers with real predict passes.

I'm weighing whether to fit the head as a plain linear probe by shrinking the tanh layer so it acts linearly, or to distill toward the base model's pre-tanh outputs with ridge. The pre-tanh route is 768-dimensional with only about 1000 samples, so it would need heavy regularization. A direct linear probe onto the four class logits, similar to LDA, seems simpler.

I'm noting that the head can be set to pass each class score through tanh unchanged, so the argmax over classes stays the same as a plain linear probe, with no need to keep the scaling small.

I'm thinking through how the final head could be refit: a closed-form ridge on the tanh features, either keeping the trained dense layer and refitting only the output projection, or fitting the dense layer to the base model's pre-tanh outputs first.

Now I'm weighing whether LDA or a ridge fit to one-hot targets would do, with class weighting to help macro-F1 given only about 62 governance examples. I want data on how much depth is actually needed, and I'm also considering whether the uniform-attention trick in the last kept layer could approximate mean pooling.

I'm working out whether a smaller feed-forward width is worth it: narrowing to 256 units per layer cuts a layer's cost roughly in half, but the layer count matters most, so I'm weighing how many layers I can keep while staying near the base accuracy.

I'm thinking about whether the FFN can be narrowed without running the model, using Gaussian moment approximations of the pre-activations instead of reimplementing the forward pass. The question is whether mean and covariance estimates of the input are enough to judge which units to keep.

I'm now looking at a closed-form way to rank FFN units: treating each pre-activation as Gaussian with mean and variance computed from the weights and an input-statistics proxy, rather than rerunning the model. The open choice is which hidden state to use as that proxy, since the FFN input after LayerNorm isn't exposed, so the earlier residual state is a reasonable stand-in for ranking units.

The last kept layer only needs to get the CLS position right, since that's the only output the refit head reads, so its FFN could be pruned using CLS-position statistics. Earlier layers have to keep their FFN quality across all tokens, and the unit count has to match across layers even though which units survive can differ.

I'm considering whether the attention layers can be trimmed too, and I'll keep looking at that path next.

I'm considering whether dropping a middle layer could be offset by folding a fitted linear map into the next layer's input projections, since the residual path from the layer before the cut would still feed straight into the next block's normalization.

I'm realizing the residual path can't be transformed by a linear map, since there's no weight on it to fold a fix into. Compensating a dropped layer through the attention output doesn't work either, because the softmax mixing isn't pointwise. So layer-drop compensation is limited, and I'll go check notes.md and the working folder.



## message

I'll start by checking the working folder for notes from earlier sessions, and looking at the structure helpers and practice data.

## tool_use: run_python

```json
{
 "code": "import os, json\nprint(os.listdir(WORK))\nif (WORK/'notes.md').exists(): print(open(WORK/'notes.md').read())\nprint(structure().keys() if structure() else None)\nprint(param_count())\nprint(edits())\nprint(pass_dirs())\nP = practice()\nprint(len(P), P[0]['text'][:300], P[0]['label'])\nimport collections\nprint(collections.Counter(p['label'] for p in P))\ntoks = tokenize([p['text'] for p in P])\nL = [len(t['ids']) for t in toks]\nprint(np.mean(L), np.max(L), np.percentile(L,[50,90,99]))"
}
```

