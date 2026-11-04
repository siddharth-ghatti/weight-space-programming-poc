## reasoning summary

Mean pooling still looks like the better fit here, since the large-c setting came out ahead in the earlier comparison. Evaluating content-based attention heads offline would take token-level states, and a pass over 64 texts at layer 11 wouldn't be enough to judge them. So I'm considering testing layer-0 lexical features instead, which gave the biggest gain before, by sweeping layer pairs like (1,11), (2,11), and (3,11).

I'm weighing whether to change which layer the readout comes from, such as tweaking layer 10 so the final layer carries more lexical signal. Zeroing out layer 10's contributions would basically just pass its normalized output along, so that doesn't add anything new. Skipping the middle layers would keep token identity but lose the context, so I'm now looking at whether layer 10's per-token FFN could carry lexical information forward instead.

Weighing whether a per-token map in layer 10's FFN could carry the lexical signal forward, since fitting it would need token-level data, and a 64-text pass only yields about 2000 vectors. The MLM head is another option, but copying its transform into an encoder layer isn't straightforward. Still, a gain of around 1.5% is worth some extra machinery if it holds up.

I'm weighing a way to get the final CLS feature to carry the bag-of-embeddings signal: route the layer-0 token average through a dedicated subspace of the CLS stream, using one repurposed head at layer 0 with uniform attention.

I'm working out how to make the layer-0 bag-of-embeddings signal survive to the final CLS feature. The idea is to give the CLS stream a dedicated subspace that carries the averaged embedding, and to keep the rest of the network from reading it so the original behavior stays intact.

I'm checking whether the residual-stream subspace I'd reserve for the averaged embedding would stay clean, since the attention and FFN outputs could still write into it. The concern is that a small signal in that subspace would survive only if the network mostly ignores it, which depends on how the layers' outputs project onto those dimensions.

I'm checking whether standard-basis dimensions would work for the reserved subspace, since LayerNorm's per-dimension gain is what couples everything together. If I zero out the write and read weights for those dimensions across every layer, the rest of the network's behavior shifts somewhat, but low-variance directions in BERT's hidden states should keep that damage small.

I'm working out how the per-token scaling from each LayerNorm would affect the reserved subspace, since each layer divides by a token-dependent standard deviation and that makes the signal uneven across tokens when averaged.

I'm thinking about whether the reserved subspace can be made to carry the averaged layer-0 signal to the final CLS feature. Since the same vector gets added to every token at layer 0, the mean over tokens just rescales it by a sentence-dependent factor, which might add mild noise to the final classification.

I'm weighing whether a repurposed layer-0 head could carry the averaged embedding in a reserved subspace, which means sacrificing that head's original output and some residual dimensions. The damage to the contextual features is unclear, so I'd need to measure it with cross-validation on the edited readout. I'm also considering which head to repurpose, since removing it should cost as little as possible.

I'm now looking at whether the per-token embedding could be routed into the reserved dimensions directly, since the final layer's uniform attention would average anything carried there. The raw embedding dimensions don't line up with A0·e_t, though, so I'd still need a linear map from the 768-dim embedding down to those 8 outputs.

Now I'm checking whether a layer-0 FFN could do the per-token linear write into those 8 reserved dimensions, since GELU is nonlinear but roughly linear for large positive inputs.

I'm getting the GELU neuron trick to work: with a large bias B, GELU is effectively linear in its input, so a neuron can carry a per-token linear map. Subtracting B afterward with the output bias would leave a clean write of the layer-0 signal into the reserved dimensions, at the cost of 8 FFN neurons out of 3072.

I'm choosing the 8 carried directions for the layer-0 signal by ridge-fitting on the layer-0 and layer-1 mean features, since both sit close to the input x that the FFN will see. Ridge on the layer-1 mean already gave about 93.3%, near the 93.8% from layer 0, so the choice of directions should hold up once the edited readout is fit.

Considering whether to protect the reserved dimensions across all eleven layers or to let the final layer's uniform attention average the carried directions directly, since per-token scaling from each LayerNorm would add some noise either way.

I'm now weighing which dimensions to reserve: I'd zero the read and write weights for those dimensions in the attention, FFN, and LayerNorm parameters of layers 1 through 10, leaving the final layer to handle its own case. That costs about 1% of the residual width, so the main choice is picking dimensions with small typical magnitude and light reads, and steering clear of the large outlier dimensions in BERT.

I'm checking how the residual dimensions reserved for the averaged signal would behave at layer 0, since the attention output and LayerNorm could still touch them. If I zero the attention output writes into those dimensions and let the FFN carry the per-token linear map, the reserved subspace should stay clean through the network.

I'm planning to evaluate the edited model by comparing ridge cross-validation on its final features against the baseline, which means two passes over the data, and I'm budgeting eight passes in total.

I'm considering whether to combine the final-layer readout edit with the carry modifications, but then I couldn't tell the two effects apart. Still, comparing the edited model's ridge fit against the baseline mean from pass 2 would show the net effect directly.

So the plan is three passes: read the edited model over the first 1024 items and cross-validate against the baseline, keep the carry edits only if they match or beat the baseline, then read the remaining 472 for the final fit and verify end to end. I also want to try carrying more directions than 8, perhaps the top PCA or LDA components of the layer-0 mean, with the projection fitted only on training folds to avoid leakage.

I'm considering instead keeping the embedding-level dimensions protected through the network, so that the signal already present in the residual stream carries through without needing a new write.

I'm checking whether the reserved dimensions can take the layer-0 output directly, and noting that the rotated-basis option doesn't work because the LayerNorm gains are per-axis. So I'd write the averaged signal through a layer-0 FFN into the reserved axis dimensions, fitting the projection on the mean embedding features since the true LayerNorm input isn't available.

I'm weighing whether to fit the write matrix on the layer-0 embedding means instead, since the attention output at that layer may be small enough that the embedding is a close stand-in for the FFN input. I'd rather stick with the FFN write than try to route through the layer-0 attention heads.

I'm going to test offline whether projecting the layer-0 mean onto the top-k directions, fitted only on training folds, recovers most of the gain. If k=8 captures it, I'll stick with 8 reserved dimensions.



Adding [CLS] didn't help, so the readout will use only the mean. I'm checking offline how small a residual projection of layer-0 features can be, fitting it within each CV fold.



## tool_use: run_python

```json
{
 "code": "exec(open('helpers.py').read())\nlab = labels_all(); y = lab[:1024]; folds = make_folds(1024)\nm11 = load('mean',11); m0 = load('mean',0); m1=load('mean',1)\ndef cv_combo(src, k, lam_src, lam, mode='ridge'):\n    c=0\n    for f in folds:\n        tr=np.setdiff1d(np.arange(1024),f)\n        if mode=='ridge':\n            Wf,b = ridge_fit(src[tr], y[tr], lam_src)   # 768x8\n            A = Wf\n        elif mode=='pca':\n            s = src[tr]-src[tr].mean(0); _,_,Vt=np.linalg.svd(s/ src[tr].std(0),full_matrices=False); A=(Vt[:k]/src[tr].std(0)).T\n        X = np.concatenate([m11, src@A],1)\n        Wf,b = ridge_fit(X[tr],y[tr],lam)\n        c+=((X[f]@Wf+b).argmax(1)==y[f]).sum()\n    return round(c/1024,4)\nfor src,name in [(m0,'h0'),(m1,'h1')]:\n    for ls in [0.03,0.1]:\n        print(name,'ridge8',ls,[cv_combo(src,8,ls,lam) for lam in (0.1,0.2,0.4)])\n    for k in [16,32,64]:\n        print(name,'pca',k,[cv_combo(src,k,None,lam,'pca') for lam in (0.1,0.2,0.4)])\n"
}
```

