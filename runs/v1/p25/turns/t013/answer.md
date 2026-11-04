## reasoning summary

I'm checking whether the shallow information can be recovered from M10/M11 at the sentence level, and it turns out it can't; the linear fit is only about R² 0.55. So I need some other way to carry that information forward. I'm turning to the CLS token at layer 0, whose input is a constant vector, to see if a rank-1 query change can make its attention uniform, and whether any of that leaks into the other tokens.

I'm weighing a different route: a carrier position that nothing attends to isn't available, since every position gets attended. So I'm considering putting the carrying inside the residual stream across all tokens, using dimensions that might be nearly unused. I'm also weighing shortening the deep pipeline, since layers 0 and 6 already reach about 97%, though it still means carrying information across layers.

I'm now working out how to keep the CLS token isolated as a carrier from layer 0 through layer 11. Its own processing should preserve the information, but its changed state would still shift how the other tokens attend to it. Matching its key and value exactly to the base would leave no room to carry anything, so that isolation can only hold approximately.

I'm going to measure how often tokens attend to the CLS token across layers, using one pass over 64 texts. If they rarely attend to it, its altered state barely affects the rest of the sequence, so the carrying design stays feasible. I'm also checking whether the CLS token's own processing preserves the shallow information: the residual connection carries it through the FFN and attention, and layer normalization only rescales it.

I'm weighing whether the CLS carrier's information survives the residual stream, which keeps it as a component but dilutes it with the base processing. The final readout would see a sum of the carried shallow signal and the deep features rather than a clean separation, so some of it gets lost.

I'm weighing whether LDA's 7 supervised dimensions could carry the shallow signal into the CLS token, though fitting that projection on training labels risks an overfit cascade at test time.

I'm weighing a cross-fitting problem: the final classifier would be trained on features where the LDA projection was fit to those same sentences, so it would look overly confident. Shrinkage regularization might soften that, but I can't cross-fit inside the model itself. Instead I could fit the combined classifier offline on the base features, with separate weights on M0 and M11, and then carry the class scores from M0 through CLS to the end.

I'm working out whether the carried scores can survive ten layers of CLS processing. The attention and FFN additions would project onto the same eight directions and corrupt them, unless I write the scores with enough magnitude to dominate the CLS state. But layer normalization rescales that state, so that magnitude may not hold.

I'm checking whether a constant reference component in the CLS state could keep the s0 scores from being lost to layer normalization, since normalizing the whole vector would otherwise erase their magnitude. The trouble is that a large reference term would shrink s0 relative to it, and the base processing additions would then distort what remains.

I'm considering whether the CLS token could be made inert in layers 1 through 10, for example by adding a rank-1 term to its query so it attends only to itself. That would rely on a large constant marker dominating its state, so the carried scores stay intact.

I'm working out whether a rank-1 change to the CLS query and key could make it attend only to itself, so its own state passes through mostly unchanged. The part that matters is the small carried component: after the attention output projection and value mixing, it gets transformed in ways I still need to check.

I'm running into the problem that computing the FFN Jacobian to carry the scores precisely would mean reimplementing part of the forward pass in numpy, which I'm trying to avoid. So I'm settling for approximate carrying, with a final refit on the actual edited model's features.

I'm checking whether compressing both M0 and M11 to 384 dimensions each would fit in the CLS token, since concatenating them at full size gives 1536 dimensions. Earlier, a 128-dimensional PCA of M0 added nothing on top of M11, which is odd given that the full M0 does help, so I'm wondering whether the useful signal is spread across dimensions that PCA discards.

I'm testing whether the shallow M0 signal still helps if it's compressed with LDA instead of PCA, since PCA discarded the useful directions. I'll check this by stacking: fitting ridge scores on M0, then training a final classifier on M11 plus those scores, and comparing against the concatenated version.

I'm considering a different route: have every token at layer 0 absorb a sentence-level vector built from the mean of the layer-0 values, so the deeper layers receive that shallow signal broadcast across the whole sequence rather than routed through the CLS token. That would avoid isolating the CLS token, though it perturbs how every token is processed in layers 1 through 10.

I'm wondering whether to first lock in a simple baseline, such as the basic design with a random-feature FFN or a plain linear model, and verify it with one pass before going back to the CLS-carry idea. Then I could work through the carry design with the passes that remain. Looking at it from the final CLS's side, I'm reconsidering whether to carry the layer-0 class scores s0 directly, since the final logits would just be s0 plus the layer-11 scores.

I'm checking whether the 8 class directions of M0 can be read back linearly from M11 at the sentence level, since if they can, a ridge on M11 alone might already recover s0.

I'm wondering whether the per-token carry could work at all: if every token writes its own 8-number layer-0 class signal into a few fixed residual directions, those directions would be perturbed by layers 1 through 10 but might still be partly readable at layer 11.

I'm checking whether the final logits could be built as a fixed-weight blend, A11·M11 plus λ times A0·M0, with the A0 and A11 weights fit offline. Training-set layer-0 scores would look overconfident, so I'd pick λ with cross-validation on the base features rather than refitting on the edited model.

I'm settling on a per-token carry: each token writes its layer-0 class scores, A0·h0, into a few fixed residual directions. The directions get zero mean so layer normalization leaves them alone, though layers 1 through 10 will still perturb them.

I'm working out how the per-token carry would survive layer normalization, since each token's scaling depends on its own standard deviation, which I can't know ahead of time. Attention and FFN also read and write those same directions, so the carried numbers would be perturbed as they pass through.

I'm working out which directions in the residual stream could carry the 8 numbers with the least interference. The idea is to pick directions that the attention and FFN write matrices in layers 1 through 10 barely touch, and that they read only weakly, so the carried values survive the processing.

I'm checking which residual directions the later layers barely write into or read from, by stacking the attention output and FFN output matrices of layers 1 through 10 and the query, key, value and first FFN matrices scaled by the layer-norm gains. The aim is a direction that minimizes both, so the carried numbers survive the processing.

I'm checking whether any residual directions are almost untouched by the later layers, using the singular spectrum of the write and read matrices. If some directions have very small energy relative to typical ones, carrying the eight numbers along them would stay fairly clean. The catch is that each token already has some base activity in those directions, which acts as noise, and a large carried value would also shift the layer-norm scale for that token.

I'm checking how much carried magnitude fits without disturbing the layer-norm scale, since keeping it around 20% of the hidden norm would leave the scale nearly unchanged. Then the 8 carried values would sit along directions with little base variance, and I'd see whether pooling at layer 11 can still read out s0.

I'm weighing how to carry the eight layer-0 numbers through the residual stream, with the scaling each token's layer norm applies being the main unknown. For the final step I'd combine the carried s0 with the pooled layer-11 scores and fit only a scalar weight, to limit leakage. I'm also considering whether the layer-0 FFN could write the full M0-block score in directly.

I'm sketching a simpler carry: have layer 0's FFN add a linear map of the layer-0 normalized input into the residual, so the pooled value becomes a fixed linear function of the mean of those inputs. The catch is that layer 0's attention mixes in other tokens, so the input isn't exactly the raw embedding and would need a forward pass to pin down.

I'm weighing whether an attention head could copy each token's own layer-0 state forward, since the FFN only sees post-attention activations and not the raw residual input.

I'm considering a broadcast carry instead: an attention head that averages the token states would hand every token the same mean of the layer-k states, so the mean layer-0 signal could be written into the u-directions that way.

I'm considering whether one head in layer 0 could be turned into a uniform averager, so that every token receives the same mean of the embedding states, and therefore the same carried s0. The worry is that repurposing a head would disturb that layer's normal behaviour, so I'm wondering whether the query weights could be zeroed on a head without losing its useful function.

I'm weighing whether to drop one layer-0 head to get the uniform averaging, since that would cost its normal output. Pruning a single head usually barely moves accuracy, but the later features and the fitted readout would still shift a little, so I'd want to refit only the layer-11 part and avoid letting the carried values leak into the fit.

I'm weighing whether to sacrifice a layer-0 head, one whose output is nearly constant, so that its replacement carries the broadcast signal with the least disruption. Refitting the readout on the perturbed features would let the carried values leak into the fit, so I'd lean toward leaving A0 and A11 as they were fitted on the base features.

I'm considering whether a single layer-0 head could be turned into uniform averaging, since softmax ignores a constant shift across keys and so a different trick won't help. Making that head uniform would mean sacrificing its normal output.

Sacrificing one layer-0 head to make broadcast attention is the costly part, so I'm weighing that trade-off. At layer 11, the pooled and normalized CLS state would carry the s0 component along the u-directions, which means the final readout reduces to A11·M11 plus that carried term.

I'm checking whether pooling the broadcast s0 through LayerNorm changes the bias terms, since the scale σ_m would multiply the carried value differently from the rest. The cleaner fix is to fit the readout directly on the normalized concatenated features, so the per-sentence scale is already accounted for.

I'm going to test whether leaking the in-sample s0 into the final fit actually hurts: train the layer-0 ridge on the training set, fit the final ridge on the layer-11 features plus s0, and compare test accuracy against a plain concatenated ridge. If the stacked version stays around 97.3%, the leakage concern is manageable.

I'm weighing a 64-dimensional carry instead, since one head's value vector is 64-dim and the sacrificed head's output would limit what fits. A random or whitened projection of the layer-0 states could be tested against the concatenated ridge baseline, the leaky stacked version, and a fixed joint fit to see which holds up.

I'm going to test whether whitening a 64-dimensional PCA projection of the layer-0 states helps the readout, since earlier scaling down the low-variance components may have hurt. I'll compare it against the plain concatenated ridge baseline.



M0 isn't linearly recoverable from deep features (held-out R² ≈ 0.55), so I'll test offline how well a ≤64-dim carried summary of it can work before building the layer-0 head that would pass it forward.



## tool_use: run_python

```json
{
 "code": "F=np.load('feats.npz'); y=F['y']; folds=np.load('folds.npy')\ndef stdz(X,tr): mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; return (X-mu)/sd\ndef ridge(X,Y,lam): return np.linalg.solve(X.T@X+lam*np.eye(X.shape[1]),X.T@Y)\ndef evalfn(build, lam):\n    accs=[]\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        Z=build(tr); mu=Z[tr].mean(0); Z=Z-mu\n        Y=np.eye(8)[y[tr]]-1/8\n        A=ridge(Z[tr],Y,lam); accs.append((np.argmax(Z[te]@A,1)==y[te]).mean())\n    return round(np.mean(accs),4)\nY8=np.eye(8)[y]-1/8\nfor lam0 in [30,100,300]:\n    for w in [0.5,1,2]:\n        def b(tr):\n            X0=stdz(F['M0'],tr); A0=ridge(X0[tr],Y8[tr],lam0); s0=X0@A0\n            return np.concatenate([stdz(F['M11'],tr)/np.sqrt(2), w*s0*np.sqrt(768)],1)\n        print('stack leak lam0',lam0,'w',w,[evalfn(b,l) for l in [30,100,300]])\nfor r in [8,32,64]:\n    def b(tr):\n        X0=stdz(F['M0'],tr); U,S,Vt=np.linalg.svd(X0[tr],full_matrices=False); P=X0@Vt[:r].T/S[:r]*np.sqrt(tr.sum())\n        return np.concatenate([stdz(F['M11'],tr), P],1)\n    print('whitePCA',r,[evalfn(b,l) for l in [30,100,300]])\n    rng=np.random.RandomState(0); Rm=rng.randn(768,r)/np.sqrt(r)\n    def b2(tr):\n        X0=stdz(F['M0'],tr); return np.concatenate([stdz(F['M11'],tr), X0@Rm*np.sqrt(768/ r)],1)\n    print('randproj',r,[evalfn(b2,l) for l in [30,100,300]])"
}
```

