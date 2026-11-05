## reasoning summary

The gap between training and test error suggests the model is limited by how much data it has. With 2500 tokens the test error is barely better than doing no compensation at all, and using the full 5049 helps but not enough. I also need to check whether the constant dimensions are being removed by subtracting the mean across tokens, since the error is measured against only the varying part of the residual stream. The errors look sizeable, around 17 to 30 percent of that variation at widths of 1024 to 1536.

Now I'm weighing how to get more calibration data without rerunning everything: the 32-probe-id cap per pass limits tokens, but picking the longest paragraphs could push each pass to roughly 16k tokens. I'm also wondering whether reading hidden states from several layers in one base-model pass could avoid the distortion from probing at a single layer.

I'm trying to get around the fact that the base model's own forward pass can't be reused here, since reimplementing attention in numpy is off the table. One idea is to read the layer-L output unmodified, which gives h_L, but the FFN input a stays hidden, so I'd have to invert the normalization to recover it from the direction alone. Adding a duplicate layer isn't possible, so that route is closed.

Now I'm weighing whether to sequentially refit the layers after pruning, since probing the edited model at layer L would capture the accumulated error from earlier pruning, while still using the base model's activations as the target.

Fitting on edited inputs with a base-model target would mean probing both models on the same ids, which doubles the cost; approximating the target directly as the original FFN's output on any inputs is simpler. Since more calibration data is the clear first step, the nonlinearity limits how well dropped units can be compensated linearly. I'm also considering whether adjusting the next layer's bias could help, though that can wait. Next I want to check what an OBS-style selection score would look like.

I'm going to check the token lengths next, and I'm also wondering how to implement a quick OBS-style greedy selection with a damped Gram matrix and a bias column, to compare test error at the same width.



Held-out errors run well above in-sample ones (L1 m1024: .23 train vs .30 held-out), so the compensation fit is data-limited. Since each probe pass caps at 32 ids, I'm checking the token-length distribution to target the longest paragraphs.



## tool_use: run_python

```json
{
 "code": "import numpy as np, json\nP = practice(); toks = tokenize([p['text'] for p in P])\nlens = np.array([min(len(t['ids']),512) for t in toks])\nids = [p['id'] for p in P]\no = np.argsort(-lens)\nprint('mean', lens.mean(), 'top32', lens[o[:32]].sum(), 'next32', lens[o[32:64]].sum(), 'next32', lens[o[64:96]].sum(), lens[o[96:128]].sum())\nprint(lens[o[:130:8]])\npid = set(json.load(open('probe_ids.json')))\nprint('probe ids overlap top128', len(pid & set(ids[i] for i in o[:128])))\njson.dump({'lens':lens.tolist(),'ids':ids}, open('lens.json','w'))"
}
```

