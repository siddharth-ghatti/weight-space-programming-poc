## reasoning summary

I'm weighing whether adding C11 and M11 gives a modest but real gain, around 0.2 to 0.3 percent, which could be noise but seems plausible. The set with C11 and M11 scores 98.4 against 98.08 without them. To implement it, I'd stash the C9 and M9 contributions at layer 9 and have layer 11 read C11, M11, and that stash.

I'm checking whether a uniform attention head at layer 9 could pool M9, though that would perturb C10, C11 and M11. Reading the C9 state directly is the harder part, since layer intermediates aren't observable, so I'm considering routing it through an attention head at layer 10 instead.

I'm weighing a head that makes CLS the attention target for every token, so its value gets copied into the stash dimensions everywhere, at the cost of one head at layer l+1. The alternative of reusing the existing stash without sacrificing a head doesn't quite work, since I'd still need to write into those dimensions. The total budget looks like roughly three or four passes out of eleven.

I'm now looking at whether a uniform attention head at layer 9 could handle the M9 pooling, and whether the CLS-self head could carry C9 across. Reading C9 out through the layer 9 FFN doesn't work, since its input is never directly observable, so I'm weighing that alternative.

I'm considering a simpler route: since C11 contributes most of the gain on its own, I could keep the layer 11 readout that reads C11 and M11 directly and skip the harder work of extracting C9 at an earlier layer.

I'm checking which combinations the stash needs: C11 and M11 can be read at layer 11 directly, so the stash only has to carry M0 and C9, possibly M9 too. Writing C9 into the stash would mean a CLS self-copy head at layer 9, which costs a head.

I'm looking for which existing layer-9 head already attends mostly to CLS, since reusing it would mean only swapping its value rows and output columns to carry the C9 signal into the stash.

I'm settling on whether a head at layer 9 that already attends mostly to CLS could carry the C9 signal into the stash, by swapping its value rows and output columns instead of adding a new head. Exactness doesn't matter here, since the final readout is fit on whatever the stash ends up holding. The real concern is overfitting: if the stash is built from in-sample data and the readout is refit on the same data, the readout will lean on it too heavily. I'd need to re-fit the C9 mapping inside each fold to get an honest estimate.

I'm considering forcing a head at layer 9 to attend exactly to CLS by giving it a strong key bias and a constant query, so its value carries C9 into the stash. That would make the stash predictable and simple to simulate, though it costs the head entirely, and existing heads that already attend mostly to CLS might serve the same role.

I'm estimating the gain at about 0.2 to 0.3 percent, which is within noise given the number of discordant cases, so it isn't yet convincing. The M0 stash improvement was more solid, and I'm wondering whether spending effort on tuning the ridge penalty for robustness would be a better use of time than chasing this.

I'm sketching the simplest version: keep the M0 stash at layer 0, and at layer 9 write the M9 and C9 contributions into it, using one uniform-attention head for M9 and a second head forced to attend to CLS for C9.

I'm now checking whether the C9 and M9 contributions can be folded into the existing stash at layer 9, since the CLS token's residual already carries C9 forward into C10 and C11. The layer 9 and 10 restores and the stash-dimension ablations still need to be set up before the readout, so the honest simulation has to fit the scaling of the M0 part and the layer 9 parts separately.

I'm checking whether a simple regression can separate the two stash contributions, since both scale differently per sample and only eight dimensions carry them. I'm also thinking the in-sample fit of the C9 and M9 stash is too clean, so a readout refit on the same data would lean on it too heavily and look better than it will on held-out sentences.

I'm considering fitting one joint ridge model over the C9, M9, M0, C11' and M11' features, which avoids refitting the readout on in-sample stash values. The catch is that C11' and M11' depend on the layer 9 and 10 edits, so the split weights would only be valid if those edits are settled first.

I'm settling on a sequence: first apply the layer 9 head edits with placeholder stash weights, then read out C11' and M11'. With those features in hand, I'd fit one joint ridge over C9', M0, C11' and M11' and split the learned weights afterward, writing the stash parts at layers 0 and 9 and the readout at layer 11. A quick predict pass should confirm the LayerNorm shifts from the stash dimensions are negligible.

I'm weighing the joint-split approach against the current refit, since the stash scaling varies per sample and that could skew the split weights. Simulations suggested the method holds up under 10–20% noise, so I'm inclined to accept it and work out how the readout weights on the stash dimensions should be scaled.

I'm weighing whether a fully CLS-forced head is worth the cost, or whether a head already attending mostly to CLS could serve the C9 copy instead. I'm also reconsidering the joint fit with split weights, since that would avoid overfitting the stash to the same data as the readout.

I'm weighing whether to skip the stash entirely and have the C9 signal ride along the CLS residual into C10 and C11, or to sacrifice one layer-9 head's value dimensions to copy it. Either way, the stash's contribution would have to be measured after the fact, since the copied values can't be read off directly until the edit is in place.

I'm considering a forced-CLS head whose value carries the C9 signal into the stash, which makes the stash exactly predictable. The tricky part is that any supervised choice of the projection onto those eight value dimensions leaks label information, so the joint fit could look too clean on the same data.

I'm weighing whether to force a CLS-attending head at layer 9 and a uniform head for M9, so both stash parts become exactly simulable for an honest joint fit. The payoff looks like about 0.25%, roughly four test errors, and the M9 term may not even matter given how close those ablation numbers sit.

I'm going to back up the current edits folder first so I can revert if the new approach doesn't pan out. Then I'll find a direction at layer 9 that separates the CLS token from the other tokens, starting by pulling the token-level states there.

I'm considering whether a head at layer 9 can be forced to attend only to CLS, which would make the C9 stash exactly simulable. The catch is that I don't have token-level states at layer 9, so I'd need one read pass over the 64 texts to find a CLS-detector direction.

I'm checking whether any head at layer 9 already sends most of its CLS-row attention to position 0. If one does, I could reuse it and only swap out its eight value dimensions, which would leave the rest of its behavior alone. The stash would then be close to, though not exactly, the C9 signal I want, so I'd need to look at the attention data from the base model's layer 9 to see whether that holds.



Adding C11 and M11 to the ensemble gives roughly 98.3–98.4% against 98.08% offline, a gain at the noise level. Before deciding whether to build it, I'm backing up the verified edits and checking whether a layer-9 head sends nearly all its CLS attention to CLS.



## tool_use: run_python

```json
{
 "code": "import shutil, os, numpy as np, json\nif not os.path.exists('edits_backup_S'): shutil.copytree('edits','edits_backup_S')\nprint(len(os.listdir('edits_backup_S')))\ntk=json.load(open('pass_6/tokens.json')); lens=[len(t) for t in tk]\nfor L in [9,10,11]:\n    a=np.load(f'pass_6/attention_L{L}.npy').astype(np.float64)\n    selfcls=np.array([a[i,:,0,0] for i in range(len(lens))])            # CLS query -> CLS key\n    sep=np.array([a[i,:,0,lens[i]-1] for i in range(len(lens))])\n    unif=np.array([0.5*np.abs(a[i,:,0,:lens[i]]-1/lens[i]).sum(-1) for i in range(len(lens))])\n    print('attn_L',L,'CLS->CLS mean',np.round(selfcls.mean(0),2))\n    print('        CLS->SEP mean',np.round(sep.mean(0),2))\n    print('        CLS-row TV-unif',np.round(unif.mean(0),2))"
}
```

