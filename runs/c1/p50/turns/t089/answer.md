## reasoning summary

Layers 4 and 6 dropped with full FFN lose a little on both the orig 8 and orig 9 features, and the pruned m=1024 refit drops orig 8 to 96.88 from 97.46. That's a loss of about the same size. I should keep in mind that these 1024-sample scores are noisy, since each sample moves the number by about 0.1%, so differences around 0.3 aren't reliable yet.

I'm weighing whether the pooler-anchor readout can actually match the KRR accuracy, so I want to test a feature map built on random training anchors and compare it against full kernel ridge. I also recall that the linear baseline already reached around 98% in cross-validation, which gives me a reference point.

I'm considering pruning the FFN width to 1024 across all layers with compensation, which would bring the model to about 71.7M parameters, or 63.8M if I stop after layer 8 and pool. The open question is how far I can shrink it before accuracy falls below the base. I'm also thinking about fitting the pruned layers to preserve the discriminative directions rather than the full hidden states, though that seems harder to do well.

Now I'm weighing a drift-correcting target: instead of matching the full FFN output on drifted inputs, the pruned layer would be fit so the residual sum tracks the base model's hidden states. That would mean exposing the unpruned layer on the same tokens too, which roughly doubles the forward passes, so it's a cost I'd have to justify.

I'm working out a per-token affine target for the pruned FFN: since LayerNorm removes scale and shift, I can match the base layer's output up to a per-token scalar, so the fit target becomes the base hidden state rescaled to the drifted input.

Now I'm working out how the per-token scale and shift would be estimated from the pruned layer's own pre-norm output, so the target can be expressed in terms of the base hidden state, though the gamma entries and reserved dimensions still need handling.

I'm checking whether tok64 uses the same ids as the earlier passes, since that determines whether the exposure data overlaps. The drift-correction target should help more than pure replication, but fitting on about 64 sentences risks overfitting, so I'd lean on a ridge prior toward the replication solution. Each layer would also need its own exposure pass, which for nine layers means roughly 18 passes out of the 31 I have left.

I'm thinking about whether a full redo with drift correction is worth it, since each layer would need its own exposure pass and that's roughly nine more passes. The base token states for the same 64 ids would be needed too, and I should first check whether tok64 matches the earlier passes.

Now I'm weighing a smaller target width: FFN m at 1024 versus 768 or 512 across the kept layers, with the pool layer included or not. Each choice trades parameter count against how closely the pruned model can track the base features, and I'll need to see where that starts to cost accuracy.

I'm wondering whether a drift-corrected fit is even worth the extra passes, since I can't measure its benefit until I run them. Before deciding, I want to check the token data in tok64.npz and whether its ids match the earlier exposure passes.



Dropping layers 4 and 6 costs about as much as m=1024 pruning, so I'm checking whether tok64.npz and the expo ids can support drift-corrected pruning targets next.



## tool_use: run_python

```json
{
 "code": "import json\nt=np.load('tok64.npz'); print({k:t[k].shape for k in t.files})\nprint(json.load(open('tok_ids.json'))[:5] if isinstance(json.load(open('tok_ids.json')),list) else json.load(open('tok_ids.json')).keys())\ne=json.load(open('expo_ids.json')); print(type(e), len(e), e[:5] if isinstance(e,list) else list(e.keys())[:5])\nprint(json.load(open('expo_map.json')))\nfor k in [4,10,18]:\n    m=json.load(open(f'pass_{k}/meta.json')); print(k, m['ids'][:5], len(m['ids']), m.get('structure'))\nprint(open('keep_m1024.json').read()[:200])\nf=np.load('ffn_scores.npz'); print(f.files[:10])"
}
```

