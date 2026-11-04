## reasoning summary

The dim 308 outlier roughly triples its shift by layer 9, and the per-sample variation changes nearly as much as the centered baseline does, so the damage reaches the informative part too. That makes proxy-based FFN pruning look like a poor choice, since the systematic offset may be coming from compensation errors that pile up layer by layer.

I'm dropping the heavy FFN pruning idea. Mild pruning would save some parameters, but the accuracy risk is unclear. I'm now considering dropping whole layers instead, testing one edited structure in a single pass to compare against the base M9 and M8 accuracies.

I'm weighing how to pick which layers to drop, since mean-level changes might hide token-level effects that matter downstream. Checking per-layer relative change in both the mean and CLS representations is a cheap proxy, while a token-level read would cost a pass. I'm also keeping track of the remaining budget, since I need several passes reserved for the final fit and verification.

Rather than pulling the pooler input from base reads and reimplementing the pooling layer in numpy, I'll just read the final CLS outputs directly from the installed model. That takes two passes, which fits the budget.

I'm weighing whether to build the head so it doesn't need a separate final read, but the kernel head needs per-sample pooler inputs, so reading the activations directly is the cleanest route. I'm also considering making each experiment pass a complete candidate, so one read could double as the final CLS read for the first 1024 ids, leaving only the remaining 472 to read afterward.

I'm settling on the pooling layer design and planning to rebuild it myself rather than reuse the editor's version. The concern is that the ridge scores carried in layers 2 and 6 were fit on the full practice data, so using them in CV would leak information and inflate the results.

I'm weighing how to build the head without leaking information. The editor's carried scores in layers 2 and 6 were fit on all practice labels, so using them in CV would inflate results. I'd rather compute the linear projections from the M2 and M6 reads myself, which are unaffected by later layers, and fit the head with proper cross-validation. The catch is that the final CLS depends slightly on the reserved channels through LayerNorm statistics, so a one-shot fit can't fully account for that.

I'm dropping the simple carry-channel idea and using only the kernel on M9, since the editor's carried scores were fit on all practice labels and would leak into CV. The reserved dimensions in layers 2 and 6 can stay as they are, because later layers cut them from the reads, and I'll exclude them from the kernel features at the pooling layer.

I'm sketching the pooling layer to use uniform attention with an identity value map, so the output is just a scaled average of the token states over the unmasked positions. I'll keep the reserved dimensions out of the kernel features, since they only enter through the LayerNorm statistics.

I'm setting the pooling layer's residual to a large scaling of the mean token state, with the CLS contribution kept small so the kernel sees mostly the averaged representation.

I'm thinking about the FFN units as an alternative to the fixed pooler. Their 3072 units could add kernel capacity beyond the 768 pooler features, though joint fitting through LayerNorm and the pooler gets complicated, and I'd expect a gain of only a few tenths of a percent.

I'm working out how the LayerNorm after the FFN would behave if the reserved dimensions carry the class scores. I'd zero the reserved dimensions of the first LayerNorm's output so they don't mix in the editor's carried values, then the second LayerNorm's mean is driven by the scores alone.

I'm checking whether the output of the pooling layer stays well-scaled: the shared offset gets removed by subtracting each channel's mean, and the remaining per-sample spread sits close to one after normalization, so the tanh readout should behave.

I'm weighing whether the FFN units could hold the whole kernel machine instead of just the 768 pooler features. With 3072 units against 1496 training points, the GELU features might approximate an exact kernel ridge fit, though the dot-product form isn't guaranteed to be symmetric positive semidefinite, so that's a risk I'd need to check.

I'm checking whether the FFN units could carry the kernel instead, by evaluating GELU features on the LN-normalized M9 vector against an RBF kernel ridge baseline. The snag is that the readout weights need the unit activations on training points, and those aren't directly readable because the CLS read comes after the final LayerNorm.

I'm checking whether the LayerNorm statistics can be undone from a read with the FFN zeroed, since the reserved dimensions of the normalized vector are exactly zero. That would let me recover the pre-pooler vector, and I'm working out how the mean and scale per sample enter into it.

I'm checking whether the reserved-dimension contributions can be recovered from the normalized vector, since their values depend on unknown normalization terms. Computing the FFN activations in numpy from the read would just reproduce the forward pass for the head I'm fitting, so I'm weighing whether that's a legitimate step for the classifier.

I'm noting that the pooler's tanh outputs can be read directly as the classifier's features, which avoids reimplementing the forward pass in numpy. So the plan is to install the pooler weights as kernel centers, read the pooled vectors for every sample, and fit the classifier on those with ridge. The FFN units would need a different route, since their activations aren't directly observable.

I'm settling on the pooler's 768 tanh units as the kernel features, since they can be read out directly through the pooled output and the classifier can be ridge-fit on them. I'm also considering whether smarter center selection, like k-means or class-balanced centers, or multiple bandwidths could close the gap to full kernel ridge regression.

I'm considering using the pooler's 768 units more efficiently: roughly half as linear-style units along principal directions and half as kernel units. I'll check variants offline on the normalized M9 read, then fit the classifier on the actual pooled outputs once the weights are installed.

I'm weighing whether to choose the kernel centers as the training points or as class-wise k-means centroids, since centers drawn from all data would leak test-fold points into the CV estimate. The honest route is to evaluate offline with proper center exclusion, and then for the final model use any practice points as centers.

I'm considering whether the last real layer could be modified so the CLS query attends uniformly, which would let it pool directly without a separate pooling layer.

I'm considering whether the CLS query in layer 8 could be neutralized so that one head pools uniformly, avoiding a separate pooling layer. The obstacle is that W_q is shared across all positions, so changing it for CLS would also alter the other tokens' queries.

I'm considering a single uniform head in layer 8 that would carry the linear ridge scores into reserved dimensions, but a linear readout only reaches about 96.8% here, so that isn't enough. Dedicating 64 residual dimensions to M8 projections might get closer, though it would cut those dimensions from every read and write in layers 0 through 8.

I'm weighing whether layer 8 could pool directly for the CLS token, which would be as good as the M8 features but would drop the M9 quality I was counting on, so the trade-off is roughly 98.0% against 98.6%.

Now I'm weighing which layers to drop, so I'll compare how much each layer changes the representation, using the relative shift between consecutive mean hidden states and CKA.

I'm checking which layers change the representation the least, using the relative shift between consecutive mean hidden states and CKA, since those are the likeliest ones to drop.



Pruned FFNs drift too far (dimension 308 reaches about −14 by L9), so I'm dropping heavy pruning. Next I'll measure per-layer change with CKA and relative change to pick layers to drop instead.



## tool_use: run_python

```json
{
 "code": "import numpy as np, sys; sys.path.insert(0,str(WORK))\nexec(open(WORK/'lib.py').read())\nF = np.load(WORK/'base_feats.npz')\ndef cka(A,B):\n    A=A-A.mean(0); B=B-B.mean(0)\n    return float(np.linalg.norm(A.T@B)**2/(np.linalg.norm(A.T@A)*np.linalg.norm(B.T@B)))\nfor L in range(12):\n    A=F[f'mean_hidden_L{L}'][:,KEEP]; B=F[f'mean_hidden_L{L+1}'][:,KEEP]\n    Ac=F[f'cls_hidden_L{L}'][:,KEEP]; Bc=F[f'cls_hidden_L{L+1}'][:,KEEP]\n    rel=np.linalg.norm(B-A,axis=1).mean()/np.linalg.norm(A,axis=1).mean()\n    relc=np.linalg.norm((B-B.mean(0))-(A-A.mean(0)),axis=1).mean()/np.linalg.norm(A-A.mean(0),axis=1).mean()\n    print(f'layer {L}: mean CKA {cka(A,B):.3f} rel {rel:.3f} centered-rel {relc:.3f} | cls CKA {cka(Ac,Bc):.3f}')\n"
}
```

