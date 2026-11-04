## reasoning summary

The edited model is behaving as intended, with a diff of 0.015 and only a trace of leftover contamination, and the pooled ridge CV sits around 0.96. The next step is a read pass over the remaining 472 items so the model can be fit on all 1496. Before committing to that, I'm weighing whether to change the design, such as moving to a two-stream setup, since reading 472 items under the current design and then switching would mean re-reading everything. I'm also comparing which improvement experiments are worth the six passes, starting with the option of finalizing v1 as it stands.

Weighing a switch to a two-stream design against finalizing v1 as it stands. Finalizing v1 takes one read pass, a fit, and a verification pass, which leaves four passes spare for experiments. A v2 would need two full reads plus a verify, and its gain of about half a percent is uncertain, so I'd check it on the first 1024 items before reading the rest.

I'm considering whether a nonlinear expansion through the pooler could help, but without iterative fitting it's hard to exploit, so I'll set it aside. Smoothing the features with a mean over the middle layers looks negligible, and non-uniform token weighting in the pooling attention is uncertain. The more promising direction is pooling the CLS token at an intermediate layer and routing the final layer's heads between mean pooling and CLS, so I'm working through how that would be implemented.

I'm weighing a pair of late-layer poolings into the CLS token, since the (5,10) and (6,11) combinations both reach about 0.9666 against roughly 0.96 for v1. Making the CLS query attend uniformly at layer a while leaving other tokens nearly unchanged means subtracting the CLS query's projection along a direction u from each token's query, and that's where I'm working through the details.

Checking whether the query residual from the CLS shift is small enough to still count as a reasonable pooling. The residual is roughly a fifth of a typical query norm, so attention would come out softer than uniform rather than exactly flat. That still looks like a usable content-based pooling, and the actual outputs will show whether it holds up. A cleaner option would be zeroing the queries at that layer, which I'm weighing against this approach.

I'm checking whether a CLS-only pooling could work by modifying one attention head, but a single head only carries 64 dimensions of mean information, which isn't enough. I'll focus instead on the CLS query shift, measuring the detector direction and how much it perturbs the other tokens' queries at layer 6.

Working out how a CLS-only pooling would behave: the CLS query can be shifted so it attends roughly uniformly, but the value and output projections are shared with every other token, so they have to stay as they are. That means the CLS update at the next layer is still the original output projection applied to the mean of the layer's hidden states, passed through the original LayerNorm and feed-forward block.

I'm working out what happens to the CLS token's information through the remaining layers: its attention-derived update passes through the original value/output maps, the LayerNorms, and the feed-forward block, and then keeps getting processed normally through layers a+1 to 10 before the final-layer routing. Since that later processing reshapes the signal, it's hard to predict how well the ridge will respond to it, so it's comparable to a CLS that attended uniformly at an intermediate layer.

I'm weighing whether to pool the final layer's mean into the CLS token as well, or to stay with the simpler options, since the gain from the deeper combinations looks small. The main question is whether the extra complexity is worth it given that the improvement is uncertain. Robustness across choosing layers 8, 9, or 10 may matter more than squeezing out another fraction of a percent.

I'm sketching a different approach: a uniform-attention pooling at layer 8 that leaves the other tokens' value and output maps alone, possibly with a residual path so each token keeps some of its own state.

I'm trying to get concatenation rather than a sum: if the pooling sends each token into a separate subspace, the pooled vector keeps distinct components instead of collapsing them together. But routing tokens into different subspaces that way still mixes them within each subspace, so I'm questioning whether it actually buys anything over a plain sum.

I'm going to test whether summing the layer-8 and layer-11 means works better than concatenating them, and whether separating them into subspaces helps. Given that the gain from pooling tweaks is only about half a percent, I'm inclined to stop chasing that. The bigger question is whether the CLS token at layer 12 carries complementary signal, so I'll check whether combining it with the layer-8 mean adds anything.

I'm checking whether combining the CLS state from layer 12 with the layer-8 mean would help, but that would require the CLS token to stay normal through the final layer while the mean is taken earlier, so it doesn't fit as a pooling design. I'll set that idea aside unless the combination turns out to be a large gain, and run a quick comparison of those pairings to decide.



The edited model checks out: the final [CLS] matches LN(layer 8 mean) within 0.015, and ridge CV on the pooled features is about 96%. Before running the remaining 472 items, I'm checking whether a cheap sum or [CLS] combination is worth switching to.



## tool_use: run_python

```json
{
 "code": "\nP = practice(); lab2i = {l:i for i,l in enumerate(LABELS)}\ny = np.array([lab2i[p['label']] for p in P])[:1024]\nM = {l: np.load(f'pass_2/mean_hidden_L{l}.npy').astype(np.float64) for l in [6,8,11]}\nCl = {l: np.load(f'pass_2/cls_hidden_L{l}.npy').astype(np.float64) for l in [9,11,12]}\nLN = lambda X: (X-X.mean(1,keepdims=True))/X.std(1,keepdims=True)\ndef cv_acc(X, y, lam, reps=4):\n    n,d = X.shape; accs=[]\n    for r in range(reps):\n        perm = np.random.default_rng(r).permutation(n); folds = np.array_split(perm, 5)\n        c=0\n        for f in folds:\n            tr = np.setdiff1d(np.arange(n), f)\n            mu = X[tr].mean(0); sd = X[tr].std(0)+1e-6\n            Xt = (X[tr]-mu)/sd; Xv = (X[f]-mu)/sd\n            Y = np.eye(8)[y[tr]]\n            Wt = np.linalg.solve(Xt.T@Xt + lam*np.eye(d), Xt.T@(Y-Y.mean(0)))\n            c += ((Xv@Wt).argmax(1)==y[f]).sum()\n        accs.append(c/n)\n    return round(np.mean(accs),4)\nprint('sum LN8+LN11', [cv_acc(LN(M[8])+LN(M[11]),y,l) for l in [100,200,400]])\nprint('cls12+mean8', [cv_acc(np.concatenate([Cl[12],M[8]],1),y,l) for l in [200,400,800]])\nprint('cls11+mean8', [cv_acc(np.concatenate([Cl[11],M[8]],1),y,l) for l in [200,400,800]])\n"
}
```

