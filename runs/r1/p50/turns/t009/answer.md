## reasoning summary

I'm checking whether dropping layer 4 is safe: it agrees with the refit about 95.5% of the time, close to what dropping layer 5 gave. That isn't lossless, and each layer seems to carry weight. If in-sample agreement is around 95%, test agreement could fall to roughly 88–90%, which might cost several points of accuracy. I'm weighing whether that is too risky, starting from the tolerance of about 1 point on test accuracy.

Dropping a full layer looks too costly, so I'm turning to FFN pruning instead, which might preserve fidelity better. Halving the FFN across all six layers would save about as many parameters as one full layer. I'm also noting that the refit head can partly compensate for errors. Next I'm designing a probe edit to extract the FFN input statistics for each layer.

I'm working out a probe for layer L: zeroing the FFN output and resetting the post-attention norm so the layer output recovers the normalized FFN input, which I can then map back using the learned scale and bias.

I'm settling on a probe for each layer: zeroing its FFN output and resetting the norms so the layer exposes the normalized attention output, then mapping back with the true scale and bias. That means one pass per layer, six in total, since the structure can't duplicate or reorder layers. I'm also tallying the budget, with three spent and one wasted so far.

I'm checking whether 32 ids is enough, and I'm settling on about 6,000 tokens with memory well within limits. For layer 5 only the CLS position matters, so I'll take the CLS FFN inputs for all 1,000 examples from the cached hidden states.

For the last layer, the probe with cls_hidden over all 1,000 examples gives the exact FFN input at CLS, so I can pick units in layer 5 that preserve the CLS output. I'd rather not evaluate the final part by hand in numpy, since reimplementing the forward pass is off-limits, so I'll limit myself to unit selection and linear least-squares compensation.

I'm weighing whether to avoid computing GELU activations in the sandbox, since that might count as forward computation. Closed-form math on returned activations seems allowed, so pre-activation statistics like W1·a + b1 are a possible route. I'm considering ranking units by an importance proxy built from those pre-activations and the W2 column norms.

I'm weighing FFN pruning as the alternative, and I'll stick to scoring units from the returned activations and weights rather than computing model outputs myself in numpy.

I'm weighing FFN pruning now, scoring units by their activation statistics and the norms of their output weights, and compensating for the dropped units with a least-squares fit onto the kept ones, which keeps the layer's output close to the original.

I'm thinking about compensating for the dropped FFN units by fitting the kept units' activations with ridge regularization, since roughly 6,000 tokens against 1,024 kept coefficients per output dimension risks overfitting. I'd rather reconstruct the layer's full output than only the FFN term, though they amount to the same thing because the attention output is fixed. For choosing which units to keep, I'm weighing a simple score based on activation energy times output-weight norm against a pivoted QR on the scaled contributions, which accounts for correlation between units.

I'm planning to pick 32 examples for the probe, aiming for varied labels and moderate-length texts so there are roughly 6,000 tokens. I'm also considering reading the CLS token across all 1,000 examples for each layer, since that gives the FFN inputs at the position that matters most.

I'm setting up the probe passes for each layer: keeping the layers up to L, zeroing that layer's FFN output and setting its norms to identity, then reading the token hidden states at that depth. For the final layer only the CLS position matters, so a separate 1,000-example CLS read would give the full sample for fitting the ridge compensation.

I'm turning to the probe helper next: I'll write a function that applies and removes the probe edits, and save each probe's data as I go. I'll pick the 32 ids for the probe after that.



Dropping layer 4 keeps about 95.5% agreement after a head refit, so I'll narrow the FFNs instead. I'm setting up a probe edit to read their exact inputs and sample 32 paragraphs.



## tool_use: run_python

```json
{
 "code": "helper = r'''\ndef probe_on(L):\n    pre = f\"roberta.encoder.layer.{L}.\"\n    save_edit(pre+\"attention.output.LayerNorm.weight\", np.ones(768, np.float32))\n    save_edit(pre+\"attention.output.LayerNorm.bias\", np.zeros(768, np.float32))\n    save_edit(pre+\"output.dense.weight\", np.zeros((768,3072), np.float32))\n    save_edit(pre+\"output.dense.bias\", np.zeros(768, np.float32))\n    save_edit(pre+\"output.LayerNorm.weight\", np.ones(768, np.float32))\n    save_edit(pre+\"output.LayerNorm.bias\", np.zeros(768, np.float32))\n    set_structure(layers=list(range(L+1)), ffn_keep=3072)\ndef probe_off(L):\n    pre = f\"roberta.encoder.layer.{L}.\"\n    for s in [\"attention.output.LayerNorm.weight\",\"attention.output.LayerNorm.bias\",\"output.dense.weight\",\"output.dense.bias\",\"output.LayerNorm.weight\",\"output.LayerNorm.bias\"]:\n        drop_edit(pre+s)\n'''\nopen(WORK/'probe.py','w').write(helper)\nP = practice(); toks = tokenize([x['text'] for x in P])\nL = np.array([len(t['ids']) for t in toks]); y = np.load(WORK/'y.npy')\nrng = np.random.default_rng(1)\nsel=[]\nfor k,cnt in zip(range(4),[6,8,10,8]):\n    cand = np.where((y==k)&(L>=120)&(L<=330))[0]\n    sel += list(rng.choice(cand, cnt, replace=False))\nsel = sorted(sel)\nprint(len(sel), L[sel].sum(), L[sel].max(), [P[i]['id'] for i in sel])\njson.dump([P[i]['id'] for i in sel], open(WORK/'probe_ids.json','w'))\nexec(helper)\nprobe_on(0); print(edits(), structure()['layers'])"
}
```

