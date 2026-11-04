## reasoning summary

I'm weighing whether the mean features from several layers should go into the final classifier together. The concatenated layers land around 97.5–97.9%, RBF on layer 12 reaches about 97.6% at a small lambda, and a lower lambda for the concatenation might push it a bit higher still. Now I'm looking at the first option, using only the layer 12 mean at CLS.

I'm checking whether pooling at the last layer could work: the final layer's attention would need another step after layer 12, which isn't available, so the most I can get from attention is the mean up to layer 11, which scores about 95.9%. I'm now considering a second option, where each layer's attention pools its own input at the CLS position without disturbing the other tokens.

Pooling at CLS in layer 11 only gives me the mean of hidden_states[11], so the other layers can't be captured that way. Changing the CLS hidden state would also shift how later layers treat the other tokens, since BERT's heads attend heavily to CLS. I'm now checking whether a spare subspace in the residual stream could carry a score, though the 768-dim stream seems to have no free dimensions, and I'm considering whether layer 11's FFN applied to the pooled mean could serve as a nonlinear feature.

I'm considering whether the CLS token could carry the pooled signal in a cleaner way, perhaps by feeding the layer 11 mean through the FFN as a nonlinear feature map. Changing the CLS state would ripple into later layers, though, so I'd need to check how much that disturbs the other tokens before trusting it.

I'm considering whether the CLS token's key could be made to receive a near-zero attention score from every other token, which would keep the other positions unaffected. That would need a large constant offset in the query-key product, set through the bias on the query side, applied to a distinctive direction written into the CLS state.

I'm now checking whether the CLS token's key could receive near-zero attention from every other token, which would keep the other positions unaffected, but the bias terms would also shift scores for the rest of the sequence, so it's too tangled to rely on. The gain from ~96% to ~97.5% is worth some effort, though the risk of disturbing the other tokens matters more, so I'm leaning toward keeping the non-CLS tokens untouched through every layer and looking at how to enrich the layer 11 tokens instead.

I'm considering whether the layer 11 tokens could carry earlier-layer information through the residual stream, since that might give the complementary signal the concatenation gets. Pooling at an earlier layer, like layer 7, and passing that mean through identity-like later layers would give a single strong feature around 96.5%, but it wouldn't combine several layers yet.

I'm weighing whether the CLS token could carry an accumulated pool across layers. The catch is that the other tokens attend heavily to CLS in later layers, so changing it would ripple into their representations. I'm wondering how much that matters, since some middle-to-late heads lean on [SEP] and [CLS] as attention sinks.

I'm thinking about how the CLS token's state changing at layer 7 onward would shift the other tokens' representations, which would blur the layer 8–12 mean features. Since the mean covers about 30 tokens and CLS is one of them, the perturbation might matter. A cleaner path is to accept whatever the edited model produces: read its activations with forward passes and fit the classifier closed-form on those, so I don't need the edited features to match the base ones.

I'm thinking about compressing the multi-layer signal into the final CLS state: project each layer's mean onto a few discriminative directions, say around 8 per layer, so 13 layers fit within the 768 dimensions.

I'm running through another way to pool at layer 11: letting the CLS position's attention draw on different layer sources, but the only states available at that point are the layer-11 token states, which mix information from all earlier layers.

I'm now checking whether the CLS token could act as a free accumulator across layers, which depends on how much the other tokens actually attend to CLS in base BERT. Only the CLS row feeds the pooler, so from layer 11 onward nothing else matters, but below that every position affects the result.

I'm recalling that base BERT's attention to [CLS] is modest in most layers, with [SEP] and punctuation absorbing more. If CLS changes at some layer, its value shifts, but a normalized state with similar statistics should keep that shift moderate. I'm also looking at a way to make CLS unattractive as a key, by giving it a direction that other tokens' queries don't match, though that depends on the query-side terms.

I'm weighing whether to avoid disturbing the other tokens at all, since the attention scores would also shift from the bias terms. Measuring the edited model's features directly, and fitting the classifier on those, seems more reliable than trying to keep CLS untouched. CV on the fixed features gives an honest estimate, provided the architecture choices don't also get tuned on the same practice items.

I'm weighing whether to leave the CLS token alone and instead rewrite the layer 11 attention so each position averages uniformly with a scaled output. That would give CLS a roughly mean-pooled state after normalization, while the FFN could then act on it as a nonlinear feature map.

I'm sketching a second design where the CLS state accumulates information from layer k onward, and I'm checking what the base CLS processing looks like before I set the attention and value terms at that layer.

I'm asking whether the CLS token could keep accumulating across layers from some point onward, which would require every later layer to pool uniformly for the CLS query while the other queries attend normally. Uniform attention for the CLS row would need its query to be near zero, but that query comes from shared weights and the CLS state, so I can't set it independently without changing the other tokens.

I'm still weighing whether to alter the CLS query so pooling stays uniform, but changing the shared weights would also shift the other tokens, so I'm checking how much CLS's state differs along a direction the other tokens barely use.

I'm considering whether a large constant component could make the CLS state effectively fixed after layer normalization, so the attention and FFN updates stop mattering much. The LN output has a fixed norm, though, so the new contributions would still shift it some.

I'm considering whether to make the layer 11 CLS state uniformly pooled, but changing shared weights would also shift the other tokens, so I'm thinking about a head dedicated to that pooling instead.

I'm considering repurposing one head per layer from layer k up to 11: zeroing its query so it attends uniformly, and using its value path to carry a projection of the mean-pooled state. The open question is where its output should be written in the residual stream, since that spot would still touch every token.

I'm considering a position-based trick: the CLS query would attend uniformly to get the pooled vector, while other queries would need to avoid picking it up. Since the pooled vector is identical for every query under uniform attention, this shift would reach every token, which is the problem I need to solve.

I'm looking at whether the position-0 features could tell the CLS query apart from the others, though those features get mixed in deeper layers. Adding a per-sentence constant shift to every token might be a small enough perturbation to accept, since I'd fit the classifier on the edited features anyway. The worry is that the CLS state, renormalized and transformed by the FFN at each layer, makes the accumulated signal messy, though it may still stay informative.

I'm now wondering whether the layer-11 features alone could serve as a nonlinear readout, comparing an RBF kernel on the layer-11 pooled state against the multi-layer linear setup. Next I'd check what a random-feature approximation with a few thousand units could do through the layer-11 FFN and pooler.

I'm now thinking about fitting the readout as a small two-layer network in closed form: random or data-chosen first-layer units, then a ridge fit for the second layer's outputs. The second layer's output goes through a normalization and a tanh pooler before the classifier, so the class scores would need to be scaled to survive that.

I'm weighing a simpler route: the pooler is already a 768-unit tanh layer, so I could treat it as a random-feature network, with random weights and a ridge fit for the output layer on top of the final CLS state.

I'm comparing three ways to read out the layer 11 features: a plain linear ridge, random GELU features with 3072 units plus ridge, and GELU features centered on the training points, which would act like an RBF network. The centers have to come from the training fold only during CV, and from all 1496 items for the final model. I still need to confirm that my LayerNorm reproduces the model's own normalization, with gamma set to 1 and beta to 0.

I'm checking whether a nonlinear readout on the layer 11 features gets close to the multi-layer accuracy, since that would make the single-layer design the better choice. If it does, I can leave the other tokens untouched, because only the CLS row of layer 11 gets read out.

I'm considering whether the layer 10 output could simply pass through layer 11 unchanged, so that the residual stream keeps earlier information for the CLS token, though the layer norm gammas would still reshape it.

Next I'm checking whether dropping the multi-layer design costs much: if a nonlinear readout on the layer-11 CLS state gets close to the multi-layer accuracy, I'd keep the single-layer setup and avoid perturbing the other tokens. I'll run comparisons of RBF, random-feature, and centered GELU readouts on the layer-11 features.



Pooling mean(h11) into [CLS] at layer 11 is the only robust spot, since the last layer I can mean-pool into [CLS] only sees hidden_states[11]; concatenated layer means reach about 97.5–97.9%, and an RBF kernel on layer 12's mean about 97.6%. Next I'll test offline how far nonlinear readouts on LN(mean h11) recover.



## tool_use: run_python

```json
{
 "code": "F=np.load('feats.npz'); y=F['y']; folds=np.load('folds.npy')\ndef LN(X): \n    X=X-X.mean(1,keepdims=True); return X/np.sqrt((X**2).mean(1,keepdims=True)+1e-12)\ndef gelu(x): \n    from scipy.special import erf; return 0.5*x*(1+erf(x/np.sqrt(2)))\ndef cv(featfn, lam):\n    accs=[]\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        Z=featfn(tr)\n        mu=Z[tr].mean(0); Zc=Z-mu\n        Y=np.eye(8)[y[tr]]-1/8\n        if Z.shape[1]>tr.sum():\n            K=Zc@Zc[tr].T; a=np.linalg.solve(K[tr]+lam*np.eye(tr.sum()),Y); pr=K[te]@a\n        else:\n            A=np.linalg.solve(Zc[tr].T@Zc[tr]+lam*np.eye(Z.shape[1]),Zc[tr].T@Y); pr=Zc[te]@A\n        accs.append((np.argmax(pr,1)==y[te]).mean())\n    return round(np.mean(accs),4)\nfor L in [11,12,6]:\n    X=LN(F[f'M{L}'])\n    print(L,'lin LN', [cv(lambda tr: X, l) for l in [1,3,10,30,100]])\n    def stdz(tr): mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; return (X-mu)/sd\n    print(L,'lin std', [cv(stdz, l) for l in [30,100,300]])\nX=LN(F['M11'])\nrng=np.random.RandomState(1)\nfor sc in [0.5,1,2]:\n    def rf(tr):\n        mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Xs=(X-mu)/sd\n        R=rng.randn(768,3072)*sc/np.sqrt(768); b=rng.randn(3072)*0.5\n        return np.concatenate([Xs, gelu(Xs@R+b)],1)\n    print('rf',sc,[cv(rf,l) for l in [10,30,100,300]])"
}
```

