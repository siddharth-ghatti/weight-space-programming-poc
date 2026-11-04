## reasoning summary

I'm noting that extra layers don't improve on M0, so the M0 stash scores about 98.05 versus 97.64, and combined with the kernel it reaches around 98.17. I want to test a few other reading-layer pairings like [C10, M10, M0] and [C8, M8, M0] later, since the earlier [M0, M11] run only hit 97.5 and a readout at layer 9 plus M0 looks strongest so far. Next I'll work out the stash design.

Next I'm working out the stash in layer 0: freezing the attention scores to uniform so each head averages over the real tokens, and routing a few value dimensions of head 0 through a projection of the mean hidden state, then wiring those into the output projection's stash dimensions.

I'm working out how the stash dimensions get their values in layer 0. The residual at those dimensions picks up the embedding's per-token values, which is noise for every token except CLS, since CLS embeddings are constant across sentences. So the stash read at the CLS position is clean, but the mean-pooled readout at layer 9 would mix in noise from the other tokens' stash dimensions, and I should consider excluding those dimensions from that read.

Next I'm weighing which value dimensions of head 0 to sacrifice for the stash: the ones with the smallest output-projection norm times value variance, or else the ones whose pooled output is nearly constant across sentences, so I can fold them into a fixed bias.

I'm settling the layer 0 stash: the uniform attention setup means head 0 outputs a per-sentence mean, so I'm now checking which value dimensions to mean-ablate by the variance their output contributes across sentences. Then I'll turn to how the LayerNorm parameters for the stash dimensions affect whatever reads them.

I'm now looking at which read points for the stash dimensions in layers 1 through 8 need to be zeroed and mean-ablated, since their original activations are only known at the layer inputs and not inside the LayerNorm output.

I'm working out how to pick the stash dimensions: the ones whose token states vary least relative to the weights that read them, checked against the layer 0, 6, and 11 token data and the mean and CLS states. Then I'll zero their writes in the attention and feed-forward layers 0 through 8, including the FFN bias and second-layer rows, and handle the LayerNorm parameters for those dimensions.

I'm settling on the layer-norm side of the stash: setting γ to 1 and β to 0 for those dimensions in every LayerNorm, so each stash value just gets normalized. What I'm still unsure about is how large the LayerNorm input's standard deviation gets for the CLS token, since the stash values shift with it.

I'm worried that the LayerNorm gains for the stash dimensions could blow up the stash across 18 normalizations if each one multiplies by roughly 1/σ ≈ 1.7. I'd need to set γ for those dimensions near the typical σ at each LayerNorm rather than 1, but I don't know that σ yet and would have to estimate it from the existing normal-dimension scales.

I'm weighing the LayerNorm gain for the stash dimensions: with γ set to 1, each normalization multiplies the stash by roughly 1/σ, which would compound badly across 18 layers. Setting γ to the typical value for that LayerNorm would keep the stash scale steady, but the ratio of γ to σ means the stash would still shrink relative to the residual dimensions that get fresh additions.

I'm weighing whether to measure the stash drift directly: after the edit, I'd check the CLS stash magnitudes at each layer, and if they fall off too fast I'd retune the per-layer gains from those numbers, using a 256-text subset before running the full read.

I'm weighing a simpler readout: leave the stash dimensions out of the ridge fit on the layer-9 features, or treat them as a separate block with fixed identity weights, since the common-mode offset from the LayerNorm would cancel in the argmax anyway.

I'm checking whether the CV-validated stash pipeline still holds once the edit changes C9. The earlier joint-ridge CV fit the stash in-fold and refit the readout in-fold, so refitting on practice is fine; the real run should use the same structure, with the stash built from all-data ridge logits and the readout refit on all data.

Settling on a simpler stash: I'll use the joint-ridge weights for the M0 part of the readout rather than a separate LDA step, and keep it to 8 dimensions.

I'm picking the 8 stash dimensions by how little their token states vary at layers 0 through 8, weighted by how strongly the layers read them. I'll check the token data at layers 0 and 6 and the read-weight norms, and combine them into a single score per dimension.

I'm picking the stash dimensions by looking at which ones have small outputs, avoiding the large-valued outlier dimensions, and I'm also considering which head-0 value dimensions to replace. Next I'm working out the scale of the stash contribution so it stays close to typical residual magnitudes and doesn't disturb the LayerNorm statistics.

I'm noting that the stash dimensions' contribution to the layer 9 readout is just part of the C9 features, so the ridge fit should include them alongside the M9 block, and the constant offset from layer 0 won't matter.

I'm picking which 8 dimensions of the layer 0 token states to use for the stash, and I'll need to make sure they don't overlap with the 8 low-variance channel dimensions already used for the readout. Next I'll load the token states at layers 0, 6, and 11 along with the weights, to compute the statistics for choosing both the stash and the value dimensions.



Extra layers don't beat M0 (~98.05% vs 97.64%), so I'm going with a single M0 stash in reserved residual dims. Next I'll pick the stash dims and the 8 head-0 value dims by checking token variance and read-weight norms.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nT=np.load('tok128.npz'); print(T.files, [T[k].shape for k in T.files])"
}
```

