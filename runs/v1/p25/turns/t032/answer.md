## reasoning summary

I'm checking whether the tapped CLS state at layer 7 carries the same information as M6, since it only reaches about 96.9% alone versus 97.5% with M6 plus M11'. I'm wondering if the attention output through the composite O·V matrices is full rank, given each head's V projects down to 64 dimensions.

Checking whether the composite attention output is full rank, I'm concluding the tapped CLS carries only part of M6's benefit at layer 7, since residual terms and the FFN wash out the rest. That gives roughly 0.7% total potential, so the CLS approach looks weak. I'm turning to cheaper options, like tweaking feature normalization for the readout and using the other layer-11 heads for extra pooled views.

I'm thinking about reading each head's pooled features out directly: by temporarily setting the output projection to a scaled identity, the final CLS state would hold the 12 head projections as one 768-dimensional vector, which I could read in a single forward pass.

I'm weighing whether the tapped CLS can pick up the 768-dimensional information, but the head outputs would be squeezed into 64 dimensions each. For the uniform head, the full mean from layer 11 is available directly. For the other heads I'd only see those 64-dimensional slices, so I'm considering attention patterns like keyword attention, where each class queries for its own evidence tokens.

I'm going to test the readout directly on the model rather than rely on the tapped CLS, by reading the final CLS for all 1496 sentences over two passes.

I'm working out how the readout would be wired: the final logits would be a sum over heads of each head's gain times its 64-dimensional projected pool, and I can fold each gain into that head's value weights so the whole thing is exact inside layer 11.

I'm checking how the final CLS state would look if I set the output projection to a scaled identity, so the layer-11 head outputs get read straight off after normalization. The catch is that the normalization changes scale when the readout values are wired back in, so the features I fit on won't match the implementation exactly unless I account for that.

I'm working through how to recover the per-sample normalization scale σ from the LN output, since the readout would otherwise be fit on normalized features rather than the raw concatenation. One idea is to add a known constant component to the residual so that σ can be solved from the fixed value, though the mean term complicates that.

I'm working through the sign and centering issue: once the readout is fed through the normalization, the per-sample scale and shift bend the logits away from the plain linear form. Adding a constant dimension to the residual would let me solve for that scale from a fixed reference, though the mean term still gets in the way.

I'm now settling the sign and centering: if the fitted weights are projected so they sum to zero and the constant feature takes up the remainder, the normalization scale σ cancels out and the logits match the implementation exactly up to a positive factor. Then I need to check whether the cross-validation on this constrained fit still reflects what the implemented readout would do.

I'm weighing whether keyword-style attention patterns in layer 11 could add information beyond the uniform mean, since that's the only real source of signal here. I'm also considering a tap at layer 10 with a uniform-attention trick to pull in earlier-layer information without it getting washed out.

I'm considering whether a broadcast head at layer 6 could write the class score into every token's residual, so the signal survives to layer 11 for pooling. The worry is that tokens get diluted by attention and FFN in later layers, much like the CLS path.

Now I'm weighing how the signal from an early layer fades as it passes through later ones: at layer 11 the injected CLS change has dropped to roughly 36% of its original size, and I'm considering whether tapping the same information at several layers would help, though the later taps would mostly repeat what layer 11 already has.

I'm checking whether the shallow information is really the source of the gain, since pairing M11 with M7 through M9 each lifts accuracy a bit. I'm also wondering if the readout could happen at an earlier layer, because M6 alone already reaches about 96.5%, nearly matching M11, and combining early layers like 0 and 6 gets to 97.0%.

I'm working out whether layers 8 through 11 could be made identity for the CLS token, so the readout logits survive from an earlier tap without being washed out.

I'm now looking at putting the readout at an earlier layer R and making layers R+1 through 11 identity, since the CLS state's logits in the channel dimensions survive normalization unchanged. That would let the pooled features from any layer feed the prediction without being diluted by the later layers.

I'm noting that once layers after R are made identity, the readout survives unchanged, so a tap at an earlier layer R could feed the prediction directly. Layer 6 pooling alone already edges out layer 11, and I'm wondering whether adding shallow information from layers 0 through 3 would help further.

I'm comparing pairs of shallow and deep pooled features, like M4 with M11 or M6 with M11, to see which combinations give the best accuracy. The next question is whether tapping at layer R-1 would keep more of the signal than tapping earlier.

I'm considering whether the CLS dilution could be avoided by suppressing the CLS's own attention and FFN updates in the later layers while letting the tokens keep processing. That would keep stored readout logits intact, but it would need a special protection scheme that I haven't worked out yet.

I'm checking how the CLS state can be protected in the later layers: a rank-1 self-attention term would make the CLS output a linear function of its own residual, so the stored signal survives. I still need the FFN side to behave the same way for the CLS token.

I'm now checking whether the CLS state can keep its stored logits through the later layers: the attention and FFN updates on that token act as a nonlinear transform, so the signal survives only approximately. Even if it does, the final linear decoder would have to be fit on features that already contain the supervised logits, which risks leaking the labels into training.

I'm weighing whether a regression from the final CLS state back to the stored logits would avoid label leakage, since it targets the logits themselves rather than the labels. Simpler still would be to make the stored signal robust by putting it in a few large-magnitude dimensions, so additive noise from later layers matters less. I also have to account for the remaining budget of 16 passes and 274 runs before deciding how far to push this.

I'm going to write up the current state in notes first, since a context restart is probably close. Then I'll pick the simplest next experiment: an early readout where the layers after the chosen layer R become identity, with no shallow carry. I'll compute the pooled-plus-CLS accuracy at each R up to 11 to see whether any single early layer beats the current 96.7%.

I'm going to score the readout at layer 7 using the CLS state from the tap at layer 6 together with the mean-pooled features from layer 7, since that's a cheap check on whether this pair beats the current best.



The 96.9% tap is short of the 97.5% pair, so the near-uniform tap captures only part of M6. I'm moving the readout into layer R with layers R+1 to 11 set to identity, so I can test tap-readout pairs with shorter gaps, and I'll save progress notes first in case the context restarts.



## tool_use: run_python

```json
{
 "code": "open('notes.md','a').write(\"\"\"\n- Exp A result (passes 8-9, featsA.npz = C7..C11', M7..M11' of layer6-tapped model): tap works locally (C7' 94.9 vs base 92.5)\n  but washes out by L11 (C11'+M11' concat 96.7 = no gain). token mean feats change ~4-5%. [C7',M11'] only 96.9; M6+M11' 97.5.\n- Idea: readout at layer R (exact readout design) + layers R+1..11 identity (Wo=0,bo=0,LN g=1 b=0, W2=0,b2=0) -> can use [C_R, M_R] for any R, or tap at k with readout at R=k+1..k+2.\n- CURRENT SAVED EDITS: layer11 readout (base C11+M11) + layer6 Wq tap (tap is harmless but useless -> should drop or rebuild readout on featsA). \n\"\"\")\nF=np.load('feats.npz'); y=F['y']; folds=np.load('folds.npy'); A=dict(np.load('featsA.npz'))\ndef stdz(X,tr): mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; return (X-mu)/sd\ndef cv(featfn, lam):\n    accs=[]\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        Z=featfn(tr); mu=Z[tr].mean(0); Zc=Z-mu\n        Y=np.eye(8)[y[tr]]-1/8\n        K=Zc@Zc[tr].T; a=np.linalg.solve(K[tr]+lam*np.eye(tr.sum()),Y); pr=K[te]@a\n        accs.append((np.argmax(pr,1)==y[te]).mean())\n    return round(np.mean(accs),4)\nfor R in [4,5,6,7,8,9,10,11]:\n    fn=lambda tr: np.concatenate([0.85*stdz(F[f'C{R}'],tr),stdz(F[f'M{R}'],tr)],1)\n    print('base [C%d,M%d]'%(R,R),[cv(fn,l) for l in [100,200,300,500]])\nfor R in [7,8]:\n    fn=lambda tr: np.concatenate([0.85*stdz(A[f'C{R}'],tr),stdz(A[f'M{R}'],tr)],1)\n    print(\"tapped [C%d',M%d']\"%(R,R),[cv(fn,l) for l in [100,200,300,500]])"
}
```

