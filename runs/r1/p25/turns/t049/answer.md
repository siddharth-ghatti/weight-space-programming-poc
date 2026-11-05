## reasoning summary

I'm sorting the token counts by kind: the first two categories each have 32 tokens and the third has 3987, so I'm guessing 0 and 1 are the special tokens and 2 is the regular ones. The relative error at layer 0 for regular tokens looks like about 20% against the pre-LN2 vector, which seems large, though the mean component is removed by layer norm, so I'm checking whether that changes the picture.

I'm checking how much the 1536 kept units explain: the relative variance loss comes out near 6.9%, versus the 3.7% the estimate predicted. I want to see whether the error is mostly a constant offset or fluctuates across tokens. Using the real FFN activations from all 4051 tokens could allow better unit selection and a least-squares refit, but with that few samples versus 1536 units the risk of overfitting is high, so a ridge-style prior is an option.

I'm weighing whether the one-layer-per-pass probe can be batched: if I alter a layer's output with identity norms and a zeroed FFN, every later layer sees the wrong input, so I can't read several layers from a single pass unless that changes.

Probes can only be read one layer per pass, since altering a layer's output corrupts every later layer's input, so the true FFN inputs are capped at 32 tokens per layer for now.

I'm comparing a few budget options for keeping FFN width and layer count, with costs from roughly 67M to 75M parameters. Option C, six layers at m=2048, cuts the most at about 11.5%, while the others save less. Since the m=1536 probe agreed with the original head at 0.922, I'm wondering whether a least-squares refit on the real activations could let me push further.

I'm working out the ridge refit for the kept units: fitting the reduced output weights against the true FFN targets, with the update regularized toward the original weights, and I'm planning to pick the penalty by cross-validating across the sampled token ids.

I'm weighing a Gaussian-moment route for the Gram matrix, but the inputs aren't Gaussian, so I'm leaning toward a data-driven Gram from the 4051 tokens with a ridge pull toward the original weights. The open question is how much a least-squares refit could actually recover, so I'm planning to test it with cross-validation by holding out 8 of the 32 ids and fitting on the other 24.

I'm checking whether the held-out error drops with a least-squares refit compared against plain top-m selection, at both m=1536 and m=2048. I'm also considering whether a data-driven importance score, based on the real activations, picks better units than the Gaussian estimate, and whether a greedy selection is worth the cost. Last, I want to confirm which token kind is the CLS token.

I'm going through the error breakdown by token kind, with the CLS and EOS-like tokens showing larger relative errors. Since the pass requires high fidelity and I have 14 passes left, I'm planning to build a least-squares refit for the FFN pruning, check it with cross-validation across token ids to pick the width, then apply it and test agreement with one read.

For the last layer, only the CLS position feeds the output, so the FFN there matters and nothing else does. With 1000 CLS samples I can refit the kept units to match the logits directly, and cross-validation should show whether that holds at any width.

I'm weighing dropping the last layer against pruning FFN width, and the layer drop looks lossy since the truncated model only agreed at 0.955. Pruning FFN to about 2048 units seems more feasible, since it saves roughly as much as removing a layer. I'll set up the least-squares experiment next, starting by identifying which token kind is the CLS token and checking the errors per id.

Now I'm matching each token id to its row in the attention mask, using the pass_2 metadata and the mask from pass_5, so the per-id grouping lines up with the true FFN inputs.



The token-kind counts look split as expected (32/32/3987), so I'm confirming that 0 and 1 are the special tokens. Next I'm testing closed-form ridge refitting of kept FFN output weights against top-m selection at m=1536, 2048 and 2560, fit on 24 paragraphs and scored on 8.



## tool_use: run_python

```json
{
 "code": "import numpy as np, json, scipy.special as sps\ndef gelu(z): return 0.5*z*(1+sps.erf(z/np.sqrt(2)))\nmask=np.load('pass_5/attention_mask.npy').astype(bool)\ngrp=np.repeat(np.arange(mask.shape[0]), mask.sum(1))\npos=np.concatenate([np.arange(n) for n in mask.sum(1)])\ntk=np.load('tok_kind.npy')\nprint('kind0 pos', np.unique(pos[tk==0]), 'kind1 pos==len-1', np.all(pos[tk==1]==mask.sum(1)-1))\nnp.save('tok_grp.npy',grp)\n"
}
```

