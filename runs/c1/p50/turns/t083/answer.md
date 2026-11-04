## reasoning summary

I'm working out a sequential method for exposing layer l's output, looking at how the attention layer norm and output layer norm would transform the hidden state if their weights are set to identity.

I'm now going through the sublayer normalization details to confirm the exposed output matches the input to the feed-forward block. Next I'm planning a ridge refit of the output weights on the sentence tokens, shrunk toward the prior. Then I'll check whether the layer-9 features still match the base model under cross-validation.

I'm weighing whether the last kept layer should serve as the pooling layer, since the CLS token at the final layer feeds the pooler. That would mean its attention gets rewritten to average the previous layer's hidden states, with only the feed-forward step still operating on the CLS token. I'm considering keeping the full layer structure with an extra pooling layer instead.

I'm considering whether the last kept layer could act as the pooling layer: its attention would compute a uniform average of the previous hidden states into the CLS position, and its feed-forward block would then run a nonlinear transform on that pooled vector. That would let the final layer produce mean-pooled features while also adding a small learned nonlinearity on top of them.

I'm weighing the parameter budget now: each layer costs about 2.37M in attention weights plus 1537 per FFN unit, so a 1024-unit FFN puts each layer near 3.9M. I'm also considering whether the pooling layer's feed-forward block could act as a kernel machine over the pooled features, though only 1024 units are available to match the anchors.

Next I'm checking whether the edited model's mean-pooled hidden features at layer 9 hold up against the base model. I'll run kernel cross-validation on the same 1024 ids, and first I need to see what the base features file contains.



Pruned layers 0–8 to m=1024 with ridge-refit output.dense, and pass_19 read the pruned model's features. Next I'll compare those features to base with the kernel CV to measure the pruning cost.



## tool_use: run_python

```json
{
 "code": "import json\nfrom lib import *\nd=np.load('base_feats.npz'); print({k:d[k].shape for k in d.files})\nm19=json.load(open('pass_19/meta.json')); print(m19.keys()); ids19=m19['ids']; print(len(ids19))\nprint(json.load(open('idsplit.json')).keys() if isinstance(json.load(open('idsplit.json')),dict) else len(json.load(open('idsplit.json'))))\nm2=json.load(open('pass_2/meta.json')); print(m2.keys(), len(m2['ids']), m2['ids'][:3], ids19[:3])\nm3=json.load(open('pass_3/meta.json')); print(len(m3['ids']))"
}
```

