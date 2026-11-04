## reasoning summary

The attention-only stack loses quality with depth, since removing FFNs entirely hurts the deeper features, and the first-1024 subset looks harder given its lower baseline numbers. That makes FFNs seem necessary, so I'd need an intermediate width. Picking units without activation data feels too crude, so I'm considering pulling activation statistics from a read pass over the per-layer hidden states, though the FFN input is still unclear to me.

I'm weighing whether the FFN input activations can be read out, since the layer output only exposes them after a LayerNorm and the FFN mixes things in. One option is to edit the model so the FFN is turned off at layer L, which would make the input recoverable up to normalization, though that only covers one configuration. Failing that, I could rank units with a weight-based score that combines the pre-activation scale with the output weight norm, possibly adjusted for bias.

I'm now weighing a more careful importance score for each unit that folds in the GELU's mean and variance under a Gaussian assumption, so units with nearly constant output can be removed by folding their contribution into the bias. The catch is that BERT hidden states have a few large outlier dimensions, so per-dimension statistics of the FFN input would be needed to make the estimate less crude.

I'm checking whether the FFN input can be pulled from a read pass, but the layer outputs only show it after LayerNorm, so I'd need a covariance estimate of the token states to approximate the outlier-heavy distribution.

I'm checking whether computing statistics from the read hidden states to score units counts as reimplementing the forward pass, and I think it doesn't, since it's just linear algebra on weights and activations. So I'll rank units by the mean and variance of their pre-activations, using the state covariance together with the weights, and then approximate the GELU's expected value and variance under a Gaussian assumption. The remaining question is how well the layer-input proxy matches the real FFN input.

I'm weighing whether the FFN input activations can be recovered from a read pass, since the outputs only expose them after LayerNorm. The attention-output LayerNorm makes the residual stream the closest proxy, so I'm considering using token-level covariance statistics to score units by their pre-activation mean and variance, then estimating the GELU's expected value and variance under a Gaussian assumption.

I'm weighing the candidate architectures against the target accuracy, noting that the kernel results on the first-1024 subset run about 0.3 points below the full set. Keeping layers 0 through 8 with a pooled readout gives a parameter count close to the k=5 full-set configuration, so the question is whether a moderate hidden width such as 768 holds up.

I'm weighing whether a radically different design could beat the straightforward layer-stack options, since the cost table shows the k=5 full-FFN and the wider-depth variants land near the same parameter budget. Keeping the full FFN on a few early layers and then a pooled readout looks like the main candidate, though dropping the deeper layers would probably cost accuracy.

Now I'm weighing whether a moderate width like 768 can hold accuracy once the readout is kernel ridge, and the parameter counts for the depth options come out roughly even. Checking whether the test score of 98.31% against the 97.8% CV estimate is meaningful depends on the test set size, which I don't know yet.

I'm going to try other readouts on the layer-1 features too, varying the kernel width, normalizing differently, or using nearest neighbors, since the jump from linear to kernel was large. I'm also wondering whether the pooler's 768 tanh units can match a kernel machine's behavior with fixed centers.

I'm now weighing whether the pooler's 768 tanh units can approximate a kernel machine by using training points as centers, and I expect the resulting kernel not to be positive semidefinite, though it should still work as features under ridge. I'd then fit the classifier ridge on those pooled features and check it with cross-validation in numpy, which seems fine since it's just my readout applied to read CLS states, though I'm still unsure whether it counts as reimplementing the forward pass.

I'm settling on the idea that the pooler's random tanh features can stand in for a kernel, with training points as centers. The resulting kernel won't be positive semidefinite, but it should still work as features for ridge. I'd then fit the classifier on pooled outputs read from the model, and use numpy for the cross-validation of the hyperparameters.

I'm checking whether random tanh features with 768 units can stand in for a kernel, and how they compare to Nyström-style centers on the layer-1 features. Before going further with readouts, I want to settle the larger structure, since that drives the parameter budget. The three candidates are an early full-FFN stack with a pooled readout, a pruned version of that, and a deeper stack at moderate width, and I'm going to test how sensitive accuracy is to FFN pruning.

I'm checking whether a wider but shallower stack could hit the accuracy target at a lower parameter count than the deeper options. The per-layer cost makes deep stacks at full width look expensive, so the real question is how much accuracy each extra layer buys. Since the test score is what matters and I can't measure it directly, I'm treating the practice cross-validation as the proxy, with the kernel readout on layers 1 and 9 at about 98.1% as the reference point.

Now I'm comparing the options by parameter count: a pool-only design, a two-layer stack with the pooler, and a three-layer variant, with the pooling layer's richer use being the key lever to reach the accuracy target.

Now I'm looking at whether the pool layer's FFN at the CLS position, with its 3072 GELU units, could act as a wide nonlinear feature map on the pooled vector before the pooler's tanh layer.

The FFN in the pool layer, with its 3072 GELU units, could serve as a kernel-style feature map on the pooled CLS vector, with the output projection carrying the class scores through to the pooler and classifier.

I'm checking whether the pool layer's 3072 GELU units can serve as the readout, with the output projection carrying the class scores into the pooler. My concern is the LayerNorm after the residual: if the score dimensions are small relative to the rest of the vector, the normalization should mostly pass them through in a roughly linear way, so I'm considering how to scale them so they aren't drowned out before the ridge readout.

I'm wondering whether attention pooling over the layer-0 token states could do better than the uniform mean, since the mean loses word order and negation cues. Max-style heads that pick out a strongly class-indicative token might capture that, but I'd need token-level states to design them and don't have those yet.

I'm wondering whether I can get the token-level class directions from about 2000 tokens per pass, and whether that's enough. Alternatively, I could run cheaper experiments on mean-hidden reads, building candidate attention heads on the pool layer and evaluating kernel CV on the CLS state, at one or two passes per design. I'm also sketching a class-evidence attention design, starting from ridge directions fit on the layer-0 mean features.

I'm working out a design for the pooling layer where a set of attention heads weight tokens by how strongly they indicate each class, using token-level scores from ridge directions fit on the layer-0 mean features. The other heads would keep a uniform mean, but only 256 dimensions fit across those four heads, so I'm wondering whether a kernel on a PCA-reduced version of the mean features can still hold the information I need.

I'm checking whether the query can be made constant by zeroing W_q, so the attention scores reduce to a fixed linear function of the token states. Setting the key weights so one head's logits follow a class direction would let that head focus on tokens with strong evidence for that class, and I still need to handle whether CLS and SEP are included in the softmax.

I'm weighing whether to set the attention scale from the token-level spread of class scores, which means one pass over 64 sentences to read layer-1 states, since the mean-only reads don't show how the scores vary across tokens. I'm also reconsidering whether a learned attention pooling could reach 97.5–98%, though without training I'd be relying on heuristics.

I'm weighing a three-layer stack with a narrower FFN, around 1024 units, against two layers at 512, since the shared width decides how many parameters each layer costs. I still need to find out how much layer 0's FFN contributes to the features, because pruning it changes what the pool layer receives.

I'm comparing layer 0's FFN contribution against attention-only layers on the first-1024 subset, where it adds only about 0.2%, so the pooling layer carries most of the signal. Full-set differences look too noisy to lean on. My read is that one or two layers with a tiny FFN width would land around 96.5–97%, which leaves the gap to the 97.8% target to be closed by smarter pooling and the readout rather than by depth.

I'm checking how well 768 tanh random features can approximate the kernel's accuracy. The plan is settling on a minimal model: either pooling alone over the embeddings, or a two-layer stack with the second layer's FFN width set to 1, then pushing accuracy through multi-head pooling and a closed-form classifier on the pooler's tanh outputs.

I'm reasoning that a pure bag-of-(token, position) model with no context can't reach the 97.8% target, since the attention weights depend only on token identity and position. Even with class-evidence weighting, I expect it to top out around 97%, which is probably short of what I need.

Next I'm checking whether a two-layer stack with the second layer's FFN width set to 1 could work, since that keeps the parameter count low while the pooling does the heavy lifting. I'm also planning to test how well a 768-unit tanh readout matches the RBF kernel's accuracy on the mean features, comparing it against the linear ridge baseline.

I'm going to test a few readouts on the mean features: random 768-unit tanh features, and Nyström-style tanh features using the training points as centers, to see whether either comes close to the RBF kernel's accuracy.



Dropping every FFN hurts the deep layers badly (kernel accuracy falls to about 89–91%), though shallow layers barely need theirs. So the leading design is layer 0 plus one pooling layer at about 29.2M params, and I'm now testing whether the pooler's 768 tanh units can match the RBF kernel using random versus data-centred features.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0, str(WORK))\nfrom cvlib import *\nR = np.load(WORK/'base_reads.npz'); y=R['y']; res=np.load(WORK/'reserved.npy')\nkeep = np.setdiff1d(np.arange(768), res)\nM = lambda L: R[f'mean_hidden_L{L}'][:,keep]\ndef zs(X): return (X-X.mean(0))/(X.std(0)+1e-6)\nY=np.eye(8)[y]\ndef feat_cv(X, fmap, lams, seeds=(0,1)):\n    out={l:[] for l in lams}\n    for s in seeds:\n        f=folds(y,5,s)\n        S={l:np.zeros((len(y),8)) for l in lams}\n        for i in range(5):\n            tr=f!=i; te=f==i\n            mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6\n            Ztr=(X[tr]-mu)/sd; Zte=(X[te]-mu)/sd\n            Ftr, Fte = fmap(Ztr, Zte, y[tr])\n            fm=Ftr.mean(0); Ftr=Ftr-fm; Fte=Fte-fm\n            G=Ftr.T@Ftr; B=Ftr.T@(Y[tr]-Y[tr].mean(0))\n            for l in lams:\n                Wt=np.linalg.solve(G+l*np.eye(G.shape[0]),B); S[l][te]=Fte@Wt\n        for l in lams: out[l].append((S[l].argmax(1)==y).mean())\n    return {l:round(float(np.mean(v)),4) for l,v in out.items()}\nrng=np.random.RandomState(0)\nX=M(1)\ndef rand_tanh(a):\n    def f(Ztr,Zte,ytr):\n        d=Ztr.shape[1]; Wr=rng.randn(768,d)/np.sqrt(d)*a; b=rng.randn(768)*0.5\n        return np.tanh(Ztr@Wr.T+b), np.tanh(Zte@Wr.T+b)\n    return f\ndef nys_tanh(a, b0=0.0):\n    def f(Ztr,Zte,ytr):\n        d=Ztr.shape[1]; idx=rng.choice(len(Ztr),768,replace=False); C=Ztr[idx]/np.sqrt(d)\n        Ctr=np.linalg.norm(C,axis=1)\n        return np.tanh(a*(Ztr@C.T)/np.sqrt(d)/Ctr + b0), np.tanh(a*(Zte@C.T)/np.sqrt(d)/Ctr+b0)\n    return f\nfor a in [0.5,1,2]:\n    print('rand', a, feat_cv(X, rand_tanh(a), [0.1,1,10]))\nfor a in [1,2,4]:\n    print('nys', a, feat_cv(X, nys_tanh(a), [0.1,1,10]))\n"
}
```

