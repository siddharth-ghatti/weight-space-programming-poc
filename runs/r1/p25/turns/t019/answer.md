## reasoning summary

I'm estimating that the linear part of the FFN variance is roughly 45 to 70 percent under a Gaussian approximation, though that ignores cross-unit correlations that could shift it either way. I'm also looking at the linear compensation from the pruned units as a summed map M with rank at most 768, and I want to see how low-rank M is in practice.

I'm stepping back to weigh what a realistic outcome looks like here: for a fine-tuned small model under tight accuracy limits, no-training pruning probably only saves something like 10-20% of parameters. I'm also considering a closed-form repair step, refitting each layer's kept FFN output weights by least squares against the base model's targets, which would recover some of the lost accuracy.

I'm now asking whether a middle layer, such as layer 3, could be dropped instead, keeping the task-specific layers 4 and 5 intact. I'm checking this against the probe results, where dropping layer 5 with a head refit already reached 95.5% agreement.

I'm planning to test layer dropping with diagonal compensation: fit per-dimension scale and offset to match the next layer's hidden states, then run passes for dropping layers 1 through 4. I also want to see whether combining drops, such as two layers, helps.

I'm folding the per-dimension scale and offset into the LayerNorm of the preceding layer, so the compensation stays inside the model's existing parameters. Dropping layer 0 is out, since the embedding LayerNorm is frozen with nothing before it to absorb the fit. Which layer to test first depends on where the task information sits, and the probes suggest it builds up in the later layers.

I'm going to test layer drops next, one configuration per pass, starting with dropping layer 2 and then layer 3 with diagonal compensation. Every pass reads all 1000 ids anyway, so I'll keep reading the full set each time. Before running those tests, I'll reset the FFN edits and the bias changes to restore the original structure.



The FFN linear share is substantial (30–73% of variance), but exploiting it fully needs true FFN inputs at about one pass per layer. I'm clearing the FFN edits and dropping middle layer 3 at full width, with diagonal-affine compensation folded into the previous LayerNorm, since probes show task information enters mostly in layers 4–5.



## tool_use: run_python

```json
{
 "code": "for e in edits(): drop_edit(e)\nmask=np.load('pass_2/attention_mask.npy').astype(bool)\ndef diag_fit(t, cls_weight=0.0):\n    a=np.load(f'pass_2/token_hidden_L{t}.npy')[mask].astype(np.float64)\n    b=np.load(f'pass_2/token_hidden_L{t+1}.npy')[mask].astype(np.float64)\n    ma,mb=a.mean(0),b.mean(0)\n    slope=((a-ma)*(b-mb)).mean(0)/(a.var(0)+1e-9)\n    c=mb-slope*ma\n    return slope,c\ndef apply_drop(t):\n    slope,c=diag_fit(t)\n    g=base(f'roberta.encoder.layer.{t-1}.output.LayerNorm.weight').astype(np.float64)\n    bb=base(f'roberta.encoder.layer.{t-1}.output.LayerNorm.bias').astype(np.float64)\n    save_edit(f'roberta.encoder.layer.{t-1}.output.LayerNorm.weight',(slope*g).astype(np.float32))\n    save_edit(f'roberta.encoder.layer.{t-1}.output.LayerNorm.bias',(slope*bb+c).astype(np.float32))\n    print('slope stats',np.percentile(slope,[5,50,95]).round(3))\napply_drop(3)\nprint(set_structure(layers=[0,1,2,4,5],ffn_keep=3072)); print(edits())"
}
```

