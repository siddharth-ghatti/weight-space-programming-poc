## reasoning summary

Layer dropping looks like it costs noticeably more than the noise on these 1024-id runs, since the base L9 model scored 97.6 there versus 98.4 with the larger set. I'm now turning to whether pruning the FFN could do better than removing whole layers.

I'm weighing whether reserving more dimensions for class scores, something like a 64-dim projection of the mean at each layer, could keep more features available for the classifier without the damage I'm seeing from dropping layers.

Truncation to content layers 0 through 8 with the pool layer gives k=10 at full width, about 95.3M parameters, which still clears the accuracy bar on the 1496-id run. Dropping one more layer to reach k=9 at 88.2M also looks workable, but going smaller than that will need FFN pruning, so I'm planning activation-based selection with least-squares refitting across the content layers and the pool layer.

I'm planning to select units per layer from base-model activations and refit with least squares, so one set of exposure passes can serve every target width. Each candidate m then only needs a single evaluation pass instead of a fresh exposure run.

I'm weighing whether to pick the longest sentences for the exposure passes, since that would give roughly 3,200 tokens per pass rather than 1,900 while keeping the class mix, and I'm checking that 64 ids is still enough for the variance-based ranking and the least-squares refit.

I'm weighing how to set up the exposure for layer 0 and the layers after it, starting from the attention output and its normalization.

Next I'm checking how the exposure edit for each layer gets reverted: the layer's saved edits come out after each read, and the base model, which already carries the earlier reserved-dimension cuts, serves as the starting point.

I'm checking how the exposure edit for each layer interacts with the LayerNorm weights in layers 2 through 10, since the earlier reserved-dimension zeroing means the normalized output may already hold zeros there. Setting the gain to 1 and the bias to 0 should still recover the correct FFN input from that normalized output. Before committing nine or more exposure passes, I'm estimating whether pruning the FFN to 1024 units could keep about 98.3% while cutting the model from roughly 95 million parameters to roughly 64 million.

Considering whether a student model built in closed form could mimic the token-level features of the 3072-unit FFN, which would sidestep the layer and width trade-offs I've been testing.

I'm still weighing whether a closed-form student could mimic the layer-9 token states, though one attention layer plus an FFN probably won't capture that contextual structure well, so I'll set that aside. I'm going with activation-based pruning, selecting the longest sentences for exposure to get more tokens per pass while keeping the class mix.

I'm going to pick the 64 longest sentences for the exposure pass, so I'm checking their token lengths first, since the maximum is probably around 80 to 100 tokens.



Dropping layers 4 and 6 costs about 0.5–0.8 points at matching depth, and naive FFN pruning did worse. I'm next trying activation-based FFN pruning with least-squares refit, collecting each layer's real FFN inputs via a temporary "exposure" read.



## tool_use: run_python

```json
{
 "code": "exec(open('lib.py').read())\nP=practice(); y=labels()\ntoks=tokenize([p['text'] for p in P])\nlens=np.array([len(t['ids']) for t in toks])\nprint('len stats', lens.min(), np.median(lens), lens.max(), lens.mean())\n# 8 longest per class\nsel=[]\nfor c in range(8):\n    idx=np.where(y==c)[0]; idx=idx[np.argsort(-lens[idx])][:8]; sel+=idx.tolist()\nsel=sorted(sel)\nprint('tokens', lens[sel].sum(), 'maxT', lens[sel].max())\nids=[P[i]['id'] for i in sel]\njson.dump(ids, open('expo_ids.json','w'))\nprint(json.dumps(ids))\n"
}
```

