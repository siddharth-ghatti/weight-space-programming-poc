## reasoning summary

I'm finding that for margins under 0.5, the base model is right 62% of the time and the second-ranked answer is right 36%. Of the flips, 19 went from right to wrong and only 2 from wrong to right, so most flips seem to hurt. That could point to pruning undoing memorized training labels, though with such small counts it could also just be noise.

The flips look like they're mostly hurting, which fits pruning erasing memorized training labels, though the counts are small enough that noise could explain some of it. On the test set that memorization effect shouldn't apply, so I expect the flips there to be closer to neutral, and I'm estimating only about 3-5% test flips at m=1536.

I'm weighing whether to spend the remaining passes on more diagnostics or on improving the L0–L2 approximations directly. The L0–L2 errors are far larger than the L3–L5 ones, so that's where any gain has to come from. More data might buy a modest improvement, perhaps 6–8%, and a sequential correction for L1, fitted against edited L0 inputs, could help further at the cost of one probe pass per layer.

I'm considering whether attention could compensate for the upstream errors. The idea is to refit the attention output projection of the next layer by least squares, so that the layer-normed residual given the edited inputs approximates what the base model produces.

I'm weighing whether the attention output projection could absorb the upstream residual error, but that only works if the error is linearly predictable from the attention context, and attention mostly mixes values across tokens rather than preserving each token's own state. So that route looks weak unless attention is focused on the token itself. The more promising target seems to be the FFN of the same layer, which would need to fit the residual gap between the base output and the edited input.

I'm weighing whether to lean on the FFN-side sequential correction or change the selection metric so it accounts for downstream influence, since attention mixes tokens and the CLS readout depends on that. Honestly the remaining gains look modest, so the real decision is which width to commit to, and I'm leaning toward m=1792 as a compromise.

I'm settling on six layers, since a k=5 variant at m=2048 comes out smaller than the 6-layer m=1536 build and every earlier layer-dropping experiment scored well below the current 0.978.

I'm weighing whether more data for L0–L2 at m=1536 is worth the passes, since it might only shave the errors a little. The alternative is sequential correction for L1 and L2 using edited L0 inputs, though the L5 result suggests the downstream FFN can't really compensate, so the main hope is that fitting on the actual shifted input distribution helps a bit.

Thinking about whether the post-LN2 objective would be better, since the layer norm strips the mean and rescales, so only some of the FFN error actually matters. I suspect it's minor, though. Then there's the CLS residual at layers 1 through 3: its R2 dropped to about .69 at this width against .86 at m=2048, and since that position feeds the classifier directly, the drop may matter more than the raw error figures suggest.

I'm checking whether the CLS errors at early layers are large in absolute terms compared with ordinary token errors. The CLS position acts as an attention sink that every token reads from, so if its early-layer state is off, that could matter more than its small variance suggests, and it might be fixable with a cheap correction.

I'm checking whether the CLS errors are small in absolute terms: its FFN input is nearly constant across inputs, so its output is roughly fixed, and the kept units only need to reproduce that one vector. The low R2 may just reflect tiny variance rather than a large error, so I want to see the raw numbers.



19 of 21 decisive flips went from right to wrong, against a base of about 65%, which fits pruning erasing memorised training detail. Next I'll compare absolute errors at the `<s>` position with the per-layer token mean to see whether early-layer attention-sink states are a cheap error source.



## tool_use: run_python

```json
{
 "code": "\nimport sys; sys.path.insert(0,str(WORK))\nfrom evalh import *\nfor k in [12,17]:\n    print('pass',k)\n    for l in range(7):\n        cb=np.load(f'pass_1/cls_hidden_L{l}.npy'); ce=np.load(f'pass_{k}/cls_hidden_L{l}.npy')\n        mb=np.load(f'pass_1/mean_hidden_L{l}.npy'); me=np.load(f'pass_{k}/mean_hidden_L{l}.npy')\n        # absolute rms error per vector, and std across items\n        print(l,'cls err',round(np.sqrt(((ce-cb)**2).sum(1).mean()),3),'cls spread',round(np.sqrt(((cb-cb.mean(0))**2).sum(1).mean()),3),'cls norm',round(np.linalg.norm(cb,axis=1).mean(),2),\n              '| mean err',round(np.sqrt(((me-mb)**2).sum(1).mean()),3),'spread',round(np.sqrt(((mb-mb.mean(0))**2).sum(1).mean()),3),\n              'meanbias err', round(np.linalg.norm((ce-cb).mean(0)),3))\n"
}
```

