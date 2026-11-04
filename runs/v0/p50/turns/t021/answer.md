## reasoning summary

The pooled random tanh features come out at 94%, below the linear baseline of 95.8% on this subset, while the RBF kernel reaches 97.5%. I suspect the standardized random features are being dominated by high-variance directions, and since the tanh is nearly linear at this width, it may reduce to a linear transform of the inputs.

I'm checking whether the RBF advantage comes from near-duplicates, since the data has standard, verbose, and concise variants of the same content that could leak across the split. A 1-NN cosine cross-validation should show whether such near-duplicates exist.

I'm weighing whether a kernel or nearest-neighbor style approach could fit the test set too, since the variants might be systematic. I'm considering how to mimic an RBF-like classifier inside the network by using tanh units centered on training points and fitting a ridge classifier over them.

I'm working out how to build a kernel-like classifier inside the network, looking at whether the layer-11 feed-forward units could hold one unit per training point and act as an RBF-style approximation.

I'm working out how the kernel-machine idea could be wired into the FFN: each training point becomes a unit whose output is thresholded similarity, and the dual coefficients from the kernel ridge solve become the output weights. I'm then routing the resulting class scores into orthogonal directions in the residual stream so the final readout can pick them up.

I'm checking whether the scores can be packed into orthogonal directions of the residual stream, so LayerNorm rescales them without changing the argmax. The mean subtraction would leave those directions alone, provided they're zero-mean. But this only works if the kernel itself is symmetric, which I haven't verified yet.

I'm checking whether a training-point similarity feature layer could reproduce the kernel predictor inside the network, with the dual coefficients from a closed-form ridge solve as output weights. The kernel matrix may not be positive semidefinite, but the ridge solution is still well defined as primal ridge over 1496 similarity features.

I'm checking whether the GELU units can reproduce a linear term: the difference GELU(x) − GELU(−x) equals x, so pairs of units could carry the linear ridge weights alongside the kernel features.

I'm weighing whether the 1496 similarity features could be wired into the FFN as GELU units, with the linear readout of the residual handled by pairs of units since GELU(x) − GELU(−x) = x. The total unit count fits within the 3072 available, so I'd need to check the CV numbers before settling on that design.

I'm wondering whether the RBF kernel ridge result, 98.2% against 96.9% for linear on the full set, will hold on the test set. The CV folds were random over items, so if paraphrased variants of the same clauses fall into different folds, the CV number may be optimistic for every model. Perhaps the test set contains variants of those same underlying clauses.

I'm checking whether the training items have close paraphrased neighbors in the embedding space, using nearest-neighbor cosine similarity, and reading the matched texts to see what those near-duplicates look like.



Random tanh features underperformed the linear baseline (94.0% against 95.8%), so I'm dropping that approach. The RBF kernel still leads at 97.5%, and I'm checking whether near-duplicate items explain that edge.



## tool_use: run_python

```json
{
 "code": "lab=np.load('feats/y.npy'); E=np.load('feats/edA_cls12.npy'); P=practice()\nmu=E.mean(0); sd=E.std(0)+1e-6; Z=(E-mu)/sd; Zn=Z/np.linalg.norm(Z,axis=1,keepdims=True)\nS=Zn@Zn.T; np.fill_diagonal(S,-1)\nnn=S.argmax(1); print('1-NN acc', (lab[nn]==lab).mean(), 'mean nn sim', S.max(1).mean())\nfor i in range(0,1496,150):\n    j=nn[i]; print(round(S[i,j],3), LABELS[lab[i]][:8], LABELS[lab[j]][:8]); print('   ',P[i]['text'][:120]); print('   ',P[j]['text'][:120])\n"
}
```

