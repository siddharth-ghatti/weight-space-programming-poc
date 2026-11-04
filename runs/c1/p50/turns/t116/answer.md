## reasoning summary

Head 0 is almost uniform in its attention, so swapping it for exact uniform attention costs far less than dropping it. I'm going to zero its query weights and bias so it attends evenly, and then carve out 8 of its value dimensions to carry the extra signal.

I'm splitting head 0's output so 8 value dimensions carry the residual signal, and the rest get a rank-56 approximation of its content map, weighted by the covariance of the layer-0 features.

I'm working out the low-rank approximation of head 0's content map, weighting by the layer-0 feature covariance so the error is measured on the real input distribution, then splitting the rank-56 factor into the output and value weights. Since attention is uniform, the value bias passes straight through and can be folded into the output bias.

I'm checking whether cutting layer 0's FFN reads from the 8 carry dimensions hurts, since those inputs change the pre-activations. I could measure the size of that contribution against the full pre-activation, or replace the per-token variation with a constant bias.

I'm handling the regression compensation for layer 0's FFN reads, fitting the RES dims from the KEEP dims so the cut can be folded into W1's remaining columns. The attention reads of h0 in layer 0 stay as they are, since they are pure embeddings with no carry yet.

I'm setting the attention LayerNorm's RES gain to 1 and its bias to 0 so those dimensions pass through unchanged, then cutting layer 0's FFN reads and writes on RES while compensating the reads by regression.

I'm zeroing the carry-head writes into RES in layers 2 and 6, and setting the RES output bias to zero there. The attention and output LayerNorm RES gains go to 1 with biases at 0, and the layer-2 carry head's original weights aren't recoverable, so I'll leave that head as the previous editor left it.

I'm reviewing the remaining layers: the reserved dimensions should already be zeroed in layers 2 through 8 of the base model. I'll leave the carry heads in layers 2 and 6 zeroed, since their original weights are gone and repurposing them isn't worth the effort.

Working out how the carry signal gets into the reserved residual dimensions: the target is the mean of the RES coordinates after the head's contribution, so the head has to cancel what layer 0 already wrote there and leave the desired value. That makes the carry a linear function of the mean of h0, with the RES coordinates subtracted out, and the value dimensions of head 0 are where that computation would go.

I'm going to scale the carried signal so its per-dimension spread is around 0.5 to 1, which is enough to stand clear of numerical noise. Downstream, the pool layer's attention can apply its own gain to the reserved dimensions, so the carry only needs to be large enough at that point. I also need to keep in mind that layer normalization adds a shared offset to those dimensions at every layer.

I'm checking how the carry signal changes in size from layer to layer. Since the pass-through gain is 1, each layer norm divides by a per-token scale, and the measured growth from layer 3 to layer 6 suggests the carried values may not shrink as I assumed.

I'm checking the carry signal's size layer by layer, since the layer norms rescale it at each step. The content perturbations from the head-0 approximation, the layer-0 and layer-1 cuts, and the carry itself will shift the downstream features slightly, which I still need to account for. I'm now weighing whether to run the drift-corrected sequential re-pruning at m=1024.

I'm weighing whether to spend the remaining passes on a sequential re-pruning at m=1024, which costs about nine exposure passes and has an uncertain payoff, or to just take the m=1536 edits that already exist. The 1024 version would need a stored baseline for reverting, and its estimated accuracy sits a bit below the base model, so I'm checking that figure against the full-set estimate before deciding.

I'm checking the estimated accuracy for the 1024-anchor version, which lands a bit under the base model once the network implementation is taken into account. The 1536-anchor version looks safer at roughly 98.3, and the parameter difference is about 8M. I'm weighing whether drift-corrected sequential re-pruning at 1024 is worth the extra passes, since it could close most of that gap.

I'm planning to install the layer 0 and 1 changes first, then re-prune layers in sequence, refitting each one against the base model's activations on the same 64 exposure examples so the drift from earlier edits gets corrected. Since the carry dimensions differ from the base there, the targets will use only the kept dimensions, and the small head-0 change in layer 0 may be partly absorbed by that layer's FFN refit.

I'm weighing whether to expose two layers in a single pass to cut the number of passes, rather than the one-pass-per-layer plan, since that would save budget for the re-pruning sequence.

I'm weighing whether the FFN input can be recovered from the next layer's hidden state by solving for it per token, since the forward map is fixed. That would mean inverting the LayerNorm and FFN together, which is a nonlinear system with about 770 unknowns per token, so it would need an iterative solver.

I'm planning the rest of the sequence: nine sequential passes for layers 0 through 8, then reading features and exposing the pool layer together, followed by installing the readout and verifying the result.

I'm planning to install the layer 0 and 1 changes with a provisional carry projection, since the final weights depend on features I haven't computed yet, and then update that projection later, which only touches the reserved dimensions.

Now I'm settling on a provisional carry projection, which I'll revise once the final features exist. Fixing the pool layer design comes next, so the layer changes stay compatible with it.

I'm setting the pool layer up as a uniform-attention block with no query weights, so the CLS token just averages the token states from layer 9. Its value and output projections map that average onto the kept dimensions, with a bias that centers it on the dataset mean.

I'm working out how the pool layer's attention can be set up so the CLS token's own state doesn't swamp the mean-pooled signal. With the scaling I have in mind, the CLS contribution would still be a sizable fraction of the total, so I'm looking at whether that leftover can be cancelled.

I'm considering whether a single attention head could cancel the CLS token's own state, but it only has 64 value dimensions, so it can't fully remove the 768-dimensional CLS contribution. A better route is to make the pool layer's output scale very large, which would shrink the CLS noise to a few percent of the total.

I'm settling the pool layer's FFN input: the CLS token's own state adds a large constant-ish component, so the attention output scale has to be big enough that this noise is negligible. The FFN input then becomes a normalized version of the centered mean-pooled signal on the kept and carry dimensions, and I'll measure the anchors from the actual activations rather than from the algebra.

I'm working out the FFN units for the pool layer: each kernel unit takes a cosine-style pre-activation against an anchor vector, so with both vectors having norm sqrt(768), the GELU output lines up with the cosine similarity. For the carried dimensions, a pair of GELU units with opposite signs gives an exactly linear pass-through, using 16 units in total, though the residual path may already carry those values.

I'm noting that the linear term can come straight from the residual path, so no extra linear units are needed, with the scale set by the attention LayerNorm gain on the reserved dimensions. The plan is to fit the output weights and the carry projection jointly, then fold the projection into the carry head and set the residual to identity.

I'm settling on the output layer's details: each class score lands in the residual dimensions, with its own intercept, and the output normalization should stay roughly fixed because the content dimensions dominate its scale. That means the class scores won't distort the argmax, though I still need to check that the common scaling holds once the scores get large.

I'm setting the pool layer's pooled output so each class score maps to one of eight dimensions, centered across classes. A small scale factor keeps the tanh from saturating, since saturated values could tie and break the argmax.

I'm checking whether the centering term needs μ9 before the attention is installed, or whether the FFN can absorb the offset instead, so the pool layer can be built first and the readout measured after.

I'm settling whether the anchor centering can ride on the attention output bias, since the non-centered cosines would all sit near 0.9 and wash out the kernel. The catch is that μ9 shifts after re-pruning, so I'm weighing whether a slightly off-center offset is good enough, or whether I should read M9 first and install the pool layer afterward.

I'm weighing whether to read μ9 first via a single 1024-id pass with the sequential structure, then install the pool layer and read the 1496 features in two passes, which would be three passes total.

I'm wondering whether the anchor centering can be handled by an offset on the attention output, since μ9 will shift after re-pruning. A small mismatch might be tolerable, and I could also recenter later by changing the output bias, though that would alter z.

Checking the carry head's provisional weights against the pool layer design, and I'm settling on one more read of the first 1024 ids after re-pruning. That read will show whether the drift correction helped, and it gives the centering mean before the pool layer goes in. After that, the pool layer is installed and its exposures are read in two passes.

I'm sketching the head 0 change in layer 0: zeroing its query weights and bias so attention goes uniform, and reshaping its value and output maps so only the kept dimensions carry the mean-pooled signal, with the residual rows left at zero.

I'm turning to the attention weights in layer 0: the head's value and output maps get reshaped so that only the kept dimensions carry the pooled signal, with a rank-56 approximation weighted by the covariance of the layer's activations. The constant bias term from the original value projection gets folded into the output bias, so the head behaves as a clean uniform pooler.

I'm checking how much the rank-56 truncation of head 0's content map loses, since with uniform attention each sentence's output is just the mean-pooled signal passed through that map. The carry dimensions get their own value rows, which read the kept dimensions with the centering offset, and their output columns write straight into the reserved residual slots.

I'm checking how the carry dimensions in head 0 get their pollution cancelled, since the pooled residual picks up the mean of the content dims. The carry output should then reduce to the centered signal times w, and I still need to confirm the scale of that projection, which I expect to be around 0.5 per dimension. Next I'll turn to the layer 0 FFN, where the residual part of its input changes from content to carry dimensions, and that change has to be compensated by a regression.

I'm compensating for the removed residual inputs in layer 0's FFN by zeroing the residual columns of W1 and folding them into the kept columns and bias through a regression fit on the base FFN inputs. The same idea applies to layer 1, where Q, K, V and the W1 residual columns get folded using regressions on the base hidden states.

I'm checking how much the layer 1 residual reads change once the carry dimensions replace the old residual content, measuring the regression fit against the base FFN inputs. I also want to confirm in code that layers 2 through 8 already have zeroed residual columns in their Q, K, V and W1 weights, and that the layer 1 output LayerNorm residual gain and bias are set as intended.

I'm running the sequential refit now, starting with layer 0: exposing that layer by setting its FFN output to zero and making both of its LayerNorms identity, then reading the normalized attention output for the 64 exposure ids.

I'm setting up the drift-corrected target for each layer: the base hidden state minus the output offset, rescaled, plus the mean of the full FFN output, with the attention input subtracted out. The residual dimensions get no FFN contribution, so their target is simply zero.

I'm checking whether the drift-corrected refit will overfit, since 64 sentences against roughly 1024×760 parameters per layer is thin data. My offline test already showed in-sample error (0.112) sitting well below out-of-sample error (0.154) at layer 0. The drift-corrected targets are noisier, so I'm leaning toward a ridge strength around 100 to 300, and still undecided on whether full correction or a half-strength mix is the safer choice.

I'm settling on alpha=1 with ridge λ=100 for layer 0, since its drift from the head-0 change and residual cuts should be small. To choose alpha at each layer, I'll hold out half the sentences, compare error against base for alpha=0 and alpha=1, then refit on all 64. After the full sequential refit, I'll run a single 1024 read and compare its features against the non-sequential pass_19 result with KRR.

I'm weighing whether m=1024 with the sequential refit will reach the base-level KRR accuracy near 98.1 on the 1024 subset. If it falls short, the fallback is the m=1536 non-sequential edits, though those would need a fresh pass once the layer 0 and layer 1 changes are in.

I'm checking which heads sit at layers 2 and 6 and whether layers 2 through 8 already have their residual columns zeroed in Q, K, V and W1.



Head 0 is already close to uniform, so I'll force it to exact uniform attention and make it the carry head; next I'm checking the reserved-dim cuts in layers 2–8 and finding the old carry heads in layers 2 and 6.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nRES=np.array([83,88,222,254,444,495,506,601])\nfor l in range(0,9):\n    P=f'bert.encoder.layer.{l}.'\n    q=W(P+'attention.self.query.weight'); k=W(P+'attention.self.key.weight'); v=W(P+'attention.self.value.weight')\n    o=W(P+'attention.output.dense.weight'); w1=W(P+'intermediate.dense.weight'); w2=W(P+'output.dense.weight')\n    qb=W(P+'attention.self.query.bias')\n    zeroq=[h for h in range(12) if np.abs(q[64*h:64*h+64]).max()==0]\n    writers=[h for h in range(12) if np.abs(o[RES][:,64*h:64*h+64]).max()>0]\n    print(l,'Qcols',np.abs(q[:,RES]).max().round(3),'Kcols',np.abs(k[:,RES]).max().round(3),'Vcols',np.abs(v[:,RES]).max().round(3),'W1cols',np.abs(w1[:,RES]).max().round(3),\n          'Orows',np.abs(o[RES]).max().round(3),'W2rows',np.abs(w2[RES]).max().round(3),'b2res',np.abs(W(P+'output.dense.bias')[RES]).max().round(3),'boRes',np.abs(W(P+'attention.output.dense.bias')[RES]).max().round(3),\n          'zeroQ heads',zeroq,'RES writers',writers)"
}
```

