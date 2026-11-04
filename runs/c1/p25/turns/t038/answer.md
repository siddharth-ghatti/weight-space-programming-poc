## reasoning summary

I'm seeing that pruning FFN units at m=1536 still costs about 0.75% at L9, dropping from 97.82 to 97.07, so this approach looks expensive. The deep path at full width sits at 95.3M parameters for roughly 98.1 CV. I'm now wondering whether the proxy-based importance scores are the real problem, not the mean-folding itself.

I'm thinking the loss might come from error accumulation across pruned layers rather than the mean-folding itself, since the readout only needs discriminative information. Looking at the numbers, the drop grows with depth: small at M4, larger at M8 and M9, which suggests the deeper layers are where the cost piles up. I'm also weighing whether a shallower stack with a better-chosen readout could recover accuracy at lower parameter count, like the M6 and M8 combinations.

I'm weighing whether a distilled readout could let the shallow model match the deep model's predictions without needing the full-depth stack. The catch is that the only data is the 1496 practice sentences, so distilling from the deep model doesn't add new information about unseen inputs.

I'm considering whether the test sentences resemble the practice ones closely enough that a kernel readout behaves like nearest-neighbor matching, so distilling wouldn't be needed. Pruning FFN units uniformly across layers seems wasteful, since some layers may tolerate heavy pruning while others need the width. I'm wondering whether to check which layers' FFNs are most critical, though each test costs a full pass.

I'm considering whether an exact least-squares compensation using probes could fix the pruning error, though collecting exact FFN inputs for 64 sentences per layer per pass looks too costly. I'm also questioning the mean-folding itself: the Gaussian assumption behind the expected GELU values might be biased, and those errors could accumulate. I'm wondering whether dropping the fold or using empirical means on the proxies would do better, and how large the fold corrections are relative to the bias.

I'm checking whether dropping particular layers costs much less than dropping others. Skipping layer 4 seems to leave accuracy essentially unchanged, while skipping layer 6 and layer 8 costs more at the deeper end.

Dropping layer 4 might be worth testing next, since skipping it seemed to cost nothing. I'm comparing the kept-layer combinations to see whether a 7-layer skip stack can close the roughly 0.7% gap to the full 9-layer model.

I'm weighing whether dropping one layer, such as layer 4, could bring the parameter count down to about 88M with little accuracy loss, though that needs a test. Each dropped layer saves roughly 7M parameters, so the realistic safe range is about 88 to 95M, a modest 13 to 20% reduction. I'm now looking at whether a more radical shallow design could capture deeper semantics more cheaply.

I'm considering whether a linear approximation of the FFN could replace the pruned units, since GELU in its identity-like regime for large positive biases behaves almost linearly, though a full-rank linear map would need many units.

I'm wondering whether a linear approximation of the FFN could stand in for the pruned units, since large positive biases push GELU into a nearly linear regime. Still, BERT's FFNs are fairly nonlinear, so a linear fit would probably capture only part of the behavior, and without good token-level data I can't trust such a fit.

I'm now asking whether the exact FFN inputs and outputs for every layer can be recovered from a single pass, since the layer-by-layer hidden states are exact but the attention step in between is what's hard to reproduce.

I'm wary of whether computing FFN activations from read states counts as forward computation inside the sandbox, which the rules forbid. Gaussian-moment proxies on pre-activations are already a borderline case, and reimplementing an attention sublayer in numpy would clearly cross the line, so I'd rather not go down that path.

I'm now weighing whether to keep the deep full-width path or favor mid-depth stacks with richer carried features, such as a 7-layer design that projects the earlier layers linearly. I'm also considering whether several pooling heads over the same hidden state could recover some of the lost accuracy.

I'm weighing whether a 9-layer stack that drops layer 4 could land near 88M parameters while keeping most of the full model's accuracy. I'm also considering a kernel readout over the deeper layers, which could reach roughly 98%, against a cheaper option that pools layers 0 through 7 and mixes in earlier linear features for about 97.7%.

I'm considering whether dropping two layers could work too, but that would need more experiments, and the measurement noise of about ±0.4% makes single-pass comparisons hard to read. I could use two passes to cover all 1496 sentences. I'm also thinking of using cosine similarity between consecutive layers' token states from the earlier pass data to spot layers that barely change the representation, since those would be the best candidates to drop.

I'm checking how much each layer changes the token representations, using the relative change between consecutive layers as a signal. Layers that barely move the representation are the best candidates to drop. I also want to see whether the earlier results still hold on the full set, where the best 9-layer configuration was about 98.1% and the start model's test score of 98.3% sets the bar.



Dropping m=1536 costs about 0.75% at layer 9, so deep FFNs must stay full-width and savings should come from skipping layers instead. Layer 4 looked free in the skip test, and I'm now measuring how much each layer changes token states to pick candidates.



## tool_use: run_python

```json
{
 "code": "mask=np.load('pass_4/attention_mask.npy').astype(bool)\nH=[np.load(f'pass_4/token_hidden_L{L}.npy')[mask] for L in range(13)]\ndef cos(a,b): return (np.sum(a*b,1)/np.linalg.norm(a,axis=1)/np.linalg.norm(b,axis=1))\nres=np.load(WORK/'reserved.npy'); keep=np.setdiff1d(np.arange(768),res)\nfor L in range(12):\n    a=H[L][:,keep]; b=H[L+1][:,keep]\n    a0=a-a.mean(1,keepdims=True); b0=b-b.mean(1,keepdims=True)\n    print(L, 'cos(hs[L],hs[L+1]) mean', round(float(cos(a0,b0).mean()),3), ' rel change', round(float(np.mean(np.linalg.norm(b0-a0,axis=1)/np.linalg.norm(a0,axis=1))),3))\n"
}
```

