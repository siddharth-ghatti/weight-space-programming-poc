## reasoning summary

The model checks out: predicted logits match the expected values to within 5e-5. I have four passes left, and I'm weighing whether a change could beat the current setup with the three I can still use for reads and verification.

I'm weighing whether a second version could beat the current setup. Two reads plus a fit would be enough, and I might spend the spare pass on verification. The strongest candidate is a two-stream design, though its gain is uncertain and the attention-sink issue in the middle layers still needs checking.

I'm considering whether a two-stream design could beat the current setup, since the attention-sink behaviour of CLS in layers 2-4 is a real obstacle. Shifting the injection point or altering CLS keys and values across those layers would disturb every other token, so I'm weighing that against the expected gain.

I'm checking whether shallow-layer information could reach the CLS token later, around layer 4, without as much damage. The gain from later injection looks modest compared with layers 0 and 1, and the shallow info seems to fade quickly. I'm also wondering whether information from the other tokens' residual streams could be read out nonlinearly instead.

I'm going to test whether a shallow-layer injection could be made less damaging by reading the deep-stream pooled features from the edited model. If the cross-validated accuracy of a concatenated classifier reaches 0.972 or better, I'd build the routing design and do the final read, which still leaves one pass for verification. If it doesn't, the already-verified v1 stays as the fallback.

I'm weighing a layer-0 hack on the CLS query so it attends uniformly, using a rank-one edit that zeroes its query and leaves the other tokens' queries shifted only by a term tied to a direction fit from the layer-0 states.

I'm working out whether the CLS token's layer-0 injection still makes sense once I consider what its state looks like after the later layers. The attention output from the rank-one edit is a fixed function of the mean of the layer-0 states, so the open question is whether the CLS representation carried to layer 8 keeps enough of that signal to be useful.

I'm wondering whether the CLS token could still carry the bag-of-words signal after several layers. Injecting the averaged layer-0 states through the attention output does shift CLS's representation at layer 1, and the query then changes with content, which would alter what it attends to. Whether that shifted state stays linearly readable after eight layers of nonlinear processing is the open question.

I'm considering a variant where CLS attends uniformly at every layer from 0 to a-1, so it accumulates running means of the layer states. That would make it a residual sum of layer means rather than a concatenation, and earlier sums of layer means did about as well as single layers, so the injected BoW signal may get diluted by later additions.

I'm comparing a few pooling options against the baseline: concatenating the CLS state at layer 9 with the layer-8 mean looks like about a +0.4% gain, while concatenating two layer means would add only a tiny amount. Using non-uniform weighting over tokens is probably not worth pursuing, since the current cross-validation already includes all tokens.

I'm weighing whether a layer-0 CLS hack is worth one pass to evaluate, since it could add about half a percent if the hacked model's CLS and mean features combine well. The catch is that I'd need a way to route keys to the CLS detector, and I can't design that from token-level reads of the hacked model.

Now I'm weighing a second experiment that might be cheaper to verify: concatenating a late-layer CLS-only pooled read with the layer-11 mean, with the attention heads split between uniform pooling and CLS self-routing.

I'm now thinking about pooling at layer 8 into just the CLS token, with the CLS-specific attention trick applied there, so the CLS keeps running through layers 9 and 10 while a uniform-pooling head set and a self-routing head set read it at layer 11.

I'm still weighing a late-layer alternative, where the CLS-only processing at layers 8-10 would produce a transformed mixture that might not match either the original CLS or the mean-pool. I can't predict its cross-validated score, so it looks too uncertain. Another possibility is to keep all tokens intact through layers 9-10 and confine the injection to a subspace, but that would perturb them heavily, so I'm leaning against it. I'm now turning to the option of using both pooling types in the final classifier.

I'm considering whether, at layer 8, the other tokens could absorb the mean of the layer states while the CLS token keeps its normal attention, so that its state at the next layer stays unchanged.

I'm checking whether, at layer 8, the CLS query can stay normal while the other tokens get a uniform query, so the CLS state at layer 9 is unchanged. The other tokens would then absorb the mean of layer-8 states through the shared value and output projections, mixing that shared component with their own individual states. I'd then read the final pooled features from layers 9 through 11.

I'm checking whether, at layer 8, the non-CLS tokens can get a uniform query while the CLS query stays normal, so the CLS state at layer 9 is unchanged. The other tokens would absorb the mean of the layer-8 states through the shared value and output projections, and I'd read the final pooled features from layer 9 onward.

I'm now looking at whether the CLS query can be approximately rewritten so it keeps its normal attention while the other tokens get a near-uniform query at layer 8. Both the CLS query error and the token leakage would be moderate, but that's acceptable because I only need the resulting features to be useful, not an exact pooling.

I'm checking whether the detector for routing keys to the CLS token still works at layer 9, since the non-CLS tokens there are already modified by the uniform attention. If some modified token scores higher on the detector than CLS does, the routing would fail.

I'm considering whether the CLS residual could carry the mean signal without attention, by scaling the attention output so the mean and CLS parts are comparable. I'd project the mean into a 384-dimensional subspace where it varies most, leaving CLS's remaining directions intact.

I'm going to mix the mean signal into the CLS part of the residual, scaling it so the mean dominates in the 384-dimensional subspace S1 while the CLS component survives in its complement. I'd then check how the combined features score offline using the base data, with the mean taken from the modified tokens at layer 9.

I'm going to test the residual-mixing idea offline on the base data, first checking whether plain concatenation of cls9 and mean9 already matches mean9 alone. If it doesn't, the design is worth a try, with the expected gain small and uncertain.

I'm going to evaluate the most promising design offline first, before committing any passes to it.



The edited model matches my logits to within 5e-5, but its 99.4% training accuracy is optimistic next to the roughly 96.8% cross-validated estimate. I'm backing up v1 as my fallback, then checking offline whether a [CLS]-plus-mean-pool design is worth an experiment with the 4 passes left.



## tool_use: run_python

```json
{
 "code": "\nimport shutil, os\nshutil.copytree('edits', 'edits_v1_backup', dirs_exist_ok=True); print(len(os.listdir('edits_v1_backup')))\nP = practice(); lab2i = {l:i for i,l in enumerate(LABELS)}\ny = np.array([lab2i[p['label']] for p in P])[:1024]\nM = {l: np.load(f'pass_2/mean_hidden_L{l}.npy').astype(np.float64) for l in [8,9,10]}\nCl = {l: np.load(f'pass_2/cls_hidden_L{l}.npy').astype(np.float64) for l in [8,9,10]}\ndef cv_acc(X, y, lam, reps=4):\n    n,d = X.shape; accs=[]\n    for r in range(reps):\n        perm = np.random.default_rng(r).permutation(n); folds = np.array_split(perm, 5)\n        c=0\n        for f in folds:\n            tr = np.setdiff1d(np.arange(n), f)\n            mu = X[tr].mean(0); sd = X[tr].std(0)+1e-6\n            Xt = (X[tr]-mu)/sd; Xv = (X[f]-mu)/sd\n            Y = np.eye(8)[y[tr]]\n            Wt = np.linalg.solve(Xt.T@Xt + lam*np.eye(d), Xt.T@(Y-Y.mean(0)))\n            c += ((Xv@Wt).argmax(1)==y[f]).sum()\n        accs.append(c/n)\n    return round(np.mean(accs),4)\nfor a in [9,10]:\n    print(f'cls{a}+mean{a}', [cv_acc(np.concatenate([Cl[a],M[a]],1),y,lam) for lam in [200,400,800]])\nprint('cls10+mean8', [cv_acc(np.concatenate([Cl[10],M[8]],1),y,lam) for lam in [200,400,800]])\n"
}
```

