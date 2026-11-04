## reasoning summary

Edited M9 comes in at 96.88% against 97.46% for the base kernel on the 1024 subset, a 0.6-point gap, and the relative error has grown to 0.44. I'm going to weigh what's achievable against the parameter budget next.

I'm comparing the parameter budgets for different depth and width setups against the fixed 24.43M base, and checking which configurations stay within reach of the 97.8% accuracy threshold.

I'm working out how the kernel's nonlinearity could come from the final layer's feed-forward block acting on the CLS position, with m units acting as anchor-like features over the pooled representation. The concern is that the closing output LayerNorm rescales the reserved class-score dimensions by a per-sentence factor, which shouldn't change the argmax but needs checking against the mean subtraction.

I'm checking whether a kernel machine on the pooled features could beat a linear readout by much. The notes don't record a linear figure for M9, so I can't tell how much of the 98.4 comes from the kernel itself. Writing the class scores into the reserved LayerNorm dimensions keeps the argmax intact, since the pooler's centering just subtracts their shared mean.

I'm sketching a final design where the pool layer is layer 11 rewritten, with its attention at the CLS position averaging over token values, so the content layers before it can be pruned to a smaller width.

I'm looking at how the attention step would produce a full 768-dimensional linear map of the mean hidden state, and whether scaling that map could let the CLS vector track the mean direction. After layer normalization, the input to the feed-forward units would then be a normalized blend of the CLS state and the mean, which would act like cosine-normalized features.

I'm working out how to make the CLS residual negligible by scaling the attention output, so that the layer normalization acts on something close to the sample-centered mean direction.

I'm working out how the attention output could be scaled so the layer normalization acts mainly on the mean hidden direction, with a bias term removing the dataset mean. Then the feed-forward units would take that normalized vector as input, and I'm checking whether a combination of cosine similarity and GELU terms can stand in for the kernel.

I'm looking at whether the anchor features could be a single GELU unit per anchor, using a scaled and shifted cosine similarity as the input, rather than two units per anchor.

I'm weighing whether the pool layer's feed-forward block can serve as the anchor features instead, with m=1024 units giving roughly 1024 anchors. If the content layers can drop to m=512, the pool layer would get 512 anchors, which means picking anchor points more carefully, perhaps with k-means centroids or PCA-direction features. I'm also considering dropping the pool layer's feed-forward entirely and letting the pooler and classifier act as a one-hidden-layer network.

The pooler's tanh layer, with 768 units, already gives a free nonlinearity, so the pool layer's feed-forward block may not be needed at all for that role.

I'm wondering whether the pooling could just be handled by the pooler itself, but it only sees the CLS token, so that won't capture the token-level information. The extra pool layer is effectively a separate layer with its own cost, so I'm asking whether the CLS vector could already carry what's needed without it.

I'm considering whether a content layer could donate a carry head to pooling, moving a 64-dimensional projection of the mean hidden state into the CLS residual stream. The previous editor only wrote 8 class scores into reserved dimensions, which is linear class evidence, so a wider reserved block might carry more, provided later layers don't overwrite it.

I'm checking whether the CLS query could be rewired in the last kept layer so that it attends uniformly, which would leave the other positions' processing alone. The catch is that query weights are shared across positions, so changing how CLS attends would also change how every other token attends.

I'm noting that the last layer only needs to serve the CLS position, so it acts as the pooling layer over whatever token states the content layers below it produce. The feature count then depends on how many content layers are kept, and I'm checking whether attention in that pool layer can pool across heads with different patterns.

I'm considering whether a carry head in an early layer could pass M0 embedding information forward, but it would only fit 8 reserved dimensions, so it works as linear scores at best. I'm wondering if that linear M0 signal, combined with the kernel on the M8 features, could recover some of the gain.

I'm weighing how much accuracy the pruned network would lose: the base 12-layer model is at 98.31% on test, against a 97.8% threshold, and the test set is probably around 1500 examples, so the sampling noise alone is about 0.36%. That means I'd want a cross-validated estimate near 98.3% before feeling safe. The pruned version looks weaker, dropping from 97.46 to 96.88 on the 1024-example subset, so the pruning itself is the main risk.

I'm comparing parameter budgets: keeping more layers at a narrower width of 1536 or 2048 costs less than the full-width alternative with fewer layers. Dropping a layer only saves a little, so narrow FFNs across more layers seem better if accuracy holds up.

I'm checking whether combining M0 with a few of the earlier content layers, like M5 or M6, gets close to the M0+M8 result, using kernel cross-validation on the full set. That would show how many content layers I actually need if the embedding information can be carried forward. The open question is how to get M0 into the final classifier nonlinearly, since the pool layer's attention only reads the mean of the last layer's tokens.

I'm wondering whether reserving more dimensions for carrying the M0 projection would help, since the 8 reserved dimensions only let me pass along a handful of numbers. I'd check how many PCA components of the embedding are needed alongside the mean features, but reserving that many dimensions would cut into what the content layers can represent.

I'm weighing whether a smarter FFN pruning than energy selection plus ridge refit could work, given a budget of about 31 forward passes and 223 code runs. The main lever I see is the width of the feed-forward layers.

I'm weighing whether fitting each layer's pruning on the base model's token states, rather than sequentially, would be cleaner, since the sequential approach needed a separate pass for every layer. The limit of 64 sentences per pass is what makes the data thin for fitting a 1024-by-768 matrix, and the ridge penalty toward the original weights only partly guards against overfitting.

I'm weighing whether unit selection could do better than keeping the highest-energy units, perhaps with a greedy or OMP-style choice that keeps the units whose removal is easiest to compensate. Kwon et al. (2022) do something close to this for BERT, combining Fisher-based mask search with least-squares tuning and reporting under 1% accuracy loss at 40% fewer FLOPs. Our aim is slightly different, since the classifier only needs to stay accurate and the hidden states need not match exactly.

Now I'm weighing whether the M9 kernel jump from roughly 97.5 at M8 to 98.4 is real, so I want to recheck per-layer kernel accuracy and the combinations on the full base features under 10-fold CV before deciding the plan. The pruned features already lose accuracy, so that comparison matters.

I'm checking how the unpruned network does with the first nine layers plus the pooling layer, since the 98.4 kernel result is the baseline to beat. I also want a linear and kernel table across layers, including the M0 combinations, and I'll confirm that the 768-anchor version of the pooled kernel matches the full kernel under cross-validation.

I'm going to tabulate per-layer linear and kernel accuracy under 10-fold CV, selecting anchors only from each training fold, to see whether the 98.4 result holds.



Pruned M9 features drop kernel accuracy about 0.6% below base on the 1024 subset (96.9 vs 97.5), which is more than the 0.5% test margin allows. I'm re-establishing the full-1496 base tables per layer and for the M0+M_l combinations next, to see how many content layers are really needed.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0, str(WORK))\nfrom lib import *\ny=labels(); d=np.load('base_feats.npz')\ndef cn(X):\n    X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.linalg.norm(X,axis=1,keepdims=True)\ndef K(Xs,g=1.0):\n    Z=np.concatenate([cn(X) for X in Xs],1)/np.sqrt(len(Xs)); return np.exp(g*(Z@Z.T-1))\nres={}\nfor l in range(13):\n    X=d[f'mean_hidden_L{l}']\n    kr=krr_cv(K([X]),y,[0.003,0.01,0.03],k=10,seeds=(0,))\n    lin=cv_acc(cn(X)*30,y,lams=(1,10),k=10)\n    k0=krr_cv(K([d['mean_hidden_L0'],X]),y,[0.003,0.01,0.03],k=10,seeds=(0,)) if l>0 else None\n    print(l,'kern',kr,'lin',lin,'M0+',k0)"
}
```

