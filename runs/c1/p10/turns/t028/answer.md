## reasoning summary

The head-of-the-original numbers put leave-one-out at 98.4%, close to the test score of 98.31%, so the LOO estimate looks only slightly optimistic, maybe by a tenth of a point. That means I should be aiming for LOO comfortably above 97.8%, ideally at least 98.3%. The shallow M0+M1 kernel only reached about 97.9% at its best grid setting, which would likely land near 97.8% on test, so it falls short of that target.

I'm weighing a kernel head on M9, which hits 98.6% LOO and looks good, against the cheaper combinations. Pairing a kernel on M8 with a linear head on M2 and M6 gives 98.1%, which is a bit low. I'm now asking whether dropping layer 7 and feeding layer 8 directly from layer 6 could match M9 at a smaller depth, though that needs a pass to confirm.

I'm considering dropping layer 7 so that layer 8 takes its input from layer 6, and I'd check the resulting depths against the base M9 and M8 numbers. I'm also wondering whether dropping two layers, 5 and 7, would save more, though if that fails I wouldn't know which drop caused it.

I'm weighing whether dropping one layer, say 5 or 7, could keep the features good enough for a classifier without any fine-tuning. Prior work on redundancy in pretrained transformers suggests middle layers can sometimes be skipped with only modest damage, though the evidence there mostly comes from fine-tuned settings. A single pass that drops one layer would tell me something useful about several of these options at once.

I'm thinking the early-layer drop might land around M8 quality, about 98.0%, though it could come out better. I also need to remember that dropping layer 6 removes its carry head, which is fine. Another option is combining features from several depths, but a pooler only produces one vector, so I'd have to split its 12 heads in some way. Using PCA-384 on M9 might capture most of what the full vector does.

I'm weighing whether a head built on the pooling layer's cls state, paired with M9, could beat the plain M9 kernel, but earlier M1 with C1 did worse than M1 alone. A two-layer model with only the embeddings and the layer-0 output feeding the pooler looks too thin to give a useful final representation, which is why I'm reconsidering how much the shallow model can actually recover.

I'm weighing whether a kernel on M1 with cosine similarity, whitened features, or a Laplacian kernel could beat the 97.9% from the M0+M1 grid, though any gain from tuning those choices would be partly selection bias.

I'm weighing whether a layer-dropped model could approach the M9 kernel's quality, and I'm wondering whether dropping layers 6 and 7 together would be worth trying, though I'd need a dedicated pass to know.

I'm weighing whether to drop one layer and still build the pooler units, but this needs a dedicated pass to confirm. The catch is that pooler centers picked from the base M9 features won't line up with a model whose pooling input has shifted.

I'm weighing whether to compute the pooler's tanh features myself in numpy from the final CLS state, rather than doing a separate read of the pooled output. That would save a pass, but it's a reimplementation of the pooler, so I'd want to check it carefully.

I'm planning the final pipeline for each config: read the final CLS states for all 1496 examples, then fit the pooler centers and classifier in closed form in numpy, and check the result with one predict pass.

I'm planning to read out the final CLS states for each config and fit the pooler and classifier in closed form, then check with one predict pass. The first exploration pass will install the pooling layer, so its reads double as the final features if that config wins. That keeps the total near seven passes.

Now I'm setting up a pooling layer at layer 11: the query weights and bias are zeroed so attention is uniform over unmasked tokens, and the value projection is the identity so each head passes through its slice of the token average.

Now I'm checking the layer-norm settings for the pooling layer: the reserved dimensions get zero gain and bias so they don't carry anything forward. The output projection is zeroed too, so the feed-forward branch adds nothing.

I'm going to check whether mixing the CLS state with the pooled mean, scaled by different weights, keeps the kernel's accuracy compared with the middle-layer representation alone. The concern is that a large scale would dilute the CLS signal, so I'll compare a few values of that weight, including the extreme where only the mean remains.

I'm writing out the pooler formulas now: scaling each feature by the dataset mean and standard deviation, then folding the centers and bias into a tanh layer that stands in for the exponential kernel.

I'm checking the pooler's tanh approximation of the exponential kernel: with a small gamma and a negative bias around -3.5, the arguments stay well below zero, so the approximation error should be under about half a percent.

I'm checking how precisely float32 can represent the pooler's tanh outputs when the arguments are very negative. With the bias near -3.5, the kernel values stay in a narrow range, and the resolution near -1 might still be fine enough for the exponential approximation.

I'm checking whether float32 precision holds up once the classifier weights get large: the constant offset of -1 in each pooled feature could be cancelled badly across hundreds of terms, so I'll look at the logit error that would result.

I'm checking whether b0 closer to -1.5 keeps the classifier stable, since with a very negative offset the constant -1 terms could cancel badly in float32. Writing the bias as c + Σβ_i with features shifted by +1 should avoid that loss of precision.

I'm keeping the pooled values in a moderate range, with the bias around -1 to -1.5, so the readout weights stay manageable. The ridge fit will need a small regularizer, which makes the weights large and the system sensitive to float32 differences between numpy and torch. So I'll check robustness by adding about 1e-5 relative noise to the features and seeing whether predictions move. A Nyström-style construction, with features built from the center kernel's inverse square root, might be a more stable alternative.

Now I'm weighing whether the pooling layer's feed-forward units, which read the layer-normalized CLS state, could push the head closer to full kernel ridge regression, and I'm checking which of those intermediate values I can actually reach.

I'm working out what the final CLS vector looks like after the pooling layer's output path: if the feed-forward output is zero, the final layer norm just normalizes the pre-norm vector, and the reserved dimensions end up set to a constant shared value. From that, I'm trying to recover the pre-norm mean, since the norm constraints (zero sum, fixed sum of squares over the 768 dimensions) should pin it down from the known features.

Now I'm checking whether I can recover the FFN activations directly: if I keep the reserved LayerNorm gains at one and zero out the feed-forward output, the final CLS state should equal the normalized input itself, so the pooler's units could be computed in numpy from what I read.

I'm worried that the reserved dimensions carry scores from the carry heads fitted on all the practice data, so letting them into the kernel would leak. I'll zero the kernel weights on those dimensions, though the layer norm statistics still include them.

I'm considering zeroing the output weights of the carry heads in layers 2 and 6, so the reserved dimensions stay at zero through the network. That would make the layer-norm statistics slightly different from the base model, but only by a small amount.

I'm noting that zeroing the carry heads' output columns would keep the reserved dimensions at zero everywhere, which slightly shifts the layer-norm statistics from the base reads but is fine since I'll read the actual final CLS state. I'm also checking whether leaving those dimensions in place could leak fitted scores into the features, and I think the only leak would be through the normalization scale, which should be negligible.

Next I'm running three offline checks: how the scale α on the mean-pooled term changes kernel LOO accuracy, whether adding linear scores from layers 2 and 6 improves on the kernel alone, and how well a 768-center realizable head does under 5-fold CV with centers taken from each training fold.



The original head design scores 98.4% leave-one-out against 98.31% on test, so LOO tracks test well and I'll require LOO of at least about 98.3%. Before building the pooling layer I'll run three offline checks: the CLS/mean mixing weight, whether M2/M6 carry-channel scores add to the L9 kernel, and how a 768-unit pooler head compares with full kernel ridge on M9 under honest CV.



## tool_use: run_python

```json
{
 "code": "import numpy as np, sys; sys.path.insert(0,str(WORK))\nexec(open(WORK/'lib.py').read())\nF = np.load(WORK/'base_feats.npz'); y=F['y']\nMf = {L: F[f'mean_hidden_L{L}'] for L in range(13)}; Cf={L: F[f'cls_hidden_L{L}'] for L in range(13)}\nM = {L: Mf[L][:,KEEP] for L in range(13)}\ndef zsall(X): mu=X.mean(0); sd=X.std(0)+1e-6; return (X-mu)/sd/np.sqrt(X.shape[1])\ndef LNr(X): X=X-X.mean(1,keepdims=True); return X/X.std(1,keepdims=True)\ndef loo(Z, y, gammas=(0.2,0.4,0.8), lams=(0.001,0.003,0.01), KL=0):\n    sq=(Z**2).sum(1); D2=np.maximum(sq[:,None]+sq[None]-2*Z@Z.T,0); T=np.eye(8)[y]-1/8; out={}\n    for g in gammas:\n        K=np.exp(-g*D2)+KL; w,V=np.linalg.eigh(K)\n        for l in lams:\n            f=w/(w+l); Hd=((V**2)*f[None]).sum(1); Yh=V@(f[:,None]*(V.T@T))\n            out[(g,l)]=round(float((((Yh-Hd[:,None]*T)/(1-Hd[:,None])).argmax(1)==y).mean()),4)\n    return out\nfor a in [0,1,2,5,20]:\n    X = LNr((a*Mf[9]+Cf[9]) if a>0 else Cf[9])[:,KEEP] if a>0 else LNr(Cf[9])[:,KEEP]\n    r=loo(zsall(X),y); print('alpha',a,max(r.values()),np.median(list(r.values())))\nX=LNr(Mf[9])[:,KEEP]; r=loo(zsall(X),y); print('pure mean LN',max(r.values()),np.median(list(r.values())))\nZL=np.concatenate([zsall(M[2]),zsall(M[6])],1)/np.sqrt(2)\nfor c in (0.03,0.1,0.3):\n    r=loo(zsall(X),y,KL=c*ZL@ZL.T); print('K9+lin26 c',c,max(r.values()),np.median(list(r.values())))\n"
}
```

