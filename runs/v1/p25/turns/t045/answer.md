## reasoning summary

I'm checking whether a compressed version works: fit the ridge on C9, M9 and M0 with nested training, and keep only M0's 8-dimensional contribution as a stash feature. The question is whether that matches the full model, which it would if the readout is a linear combination of the same pieces.

I'm working out whether a stash of M0's 8 logit contributions, fixed from the joint ridge fit, reproduces the full model exactly, since the readout is linear. I'll also try a kernel on C9 and M9 with the linear M0 stash. A worry is that stash values fit on the practice data are in-sample and could be overweighted, so I'd hold the combination weights fixed rather than refit them.

Next I'm weighing whether to fit the stash weights once on the joint model or iterate, since the stash values written at layer 0 feed back into the C9 readout. I'm also checking how zeroing reads of the stash dimensions in each layer's Q, K, V, and FFN weights would keep the rest of the computation undisturbed.

Now I'm working out how the stash dimensions get zeroed out of every read path: the Q, K, V and FFN columns, the output and second FFN rows, and the layer norm scales and offsets, so each stash dimension just passes through as normalized values.

I'm wondering whether to sacrifice one of the 12 layer-0 heads for uniform pooling, which might disturb the network more than I'd like. The alternative is to find the layer-0 head whose attention is already closest to uniform, since replacing that one would change the function least. I'd need the attention data to tell which head that is.

I'm wondering whether to sacrifice one of the layer-0 heads to get uniform pooling, or whether a head that already attends fairly evenly would be the cheaper choice. Attention data from the earlier pass should show which head that is. I could also put the stash at a later layer, where a near-uniform head might exist, though M0 carries the most signal. Either way, the downstream features will shift, so I'd re-read them and test whether the readout still holds.

I'm weighing whether to avoid sacrificing a head at all, but a free stash lane can't change the attention pattern, since a single head's softmax has to be uniform for the pooling to work. So the choice is to give up one layer-0 head and pick whichever head's current attention is closest to uniform, so that replacing it changes the function the least.

I'm weighing which layer-0 head to give up for uniform pooling, favoring the one with the highest attention entropy so the swap changes the function least. Checking the pass_6 attention data should show which head that is, and I'll also run the α-noise and kernel-plus-M0 simulations after.



M0 at the embedding layer gives a linear gain of +0.45% (98.08% vs 97.64%), so I'm considering stashing its pooled value in reserved residual dims through layers 1–8. Next I'll check for existing layer-0 attention data to pick a sacrificial head, and test how sensitive the gain is to per-sentence scale noise.



## tool_use: run_python

```json
{
 "code": "import os, json, numpy as np\nfor p in sorted(os.listdir('.')):\n    if p.startswith('pass_'):\n        print(p, os.listdir(p)[:20])\nr=json.load(open('pass_6/result.json')) if os.path.exists('pass_6/result.json') else None\nprint(r if r is None else str(r)[:1500])"
}
```

