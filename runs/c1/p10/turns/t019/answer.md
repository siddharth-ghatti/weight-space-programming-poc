## reasoning summary

I'm weighing whether a smaller c could do even better than 0.3, since the linear part seems to add a little. From the results so far, the deep model needs at least layers 0 through 7 plus pooling to reach about 98.1%, and layers 0 through 8 for roughly 98.3 to 98.5%. Next I want to look at pruning the FFN in the deeper layers, and possibly skipping some layers altogether.

I'm now looking at how to set up each experiment to be as informative as possible. A single read pass can return mean_hidden at every layer, so one run with a fixed pruning amount would give accuracy at each depth on the same 1024 samples, which makes comparison with the base easy.

I'm planning the remaining budget: three experiment passes with 1024 ids each to compare pruning amounts, then a final pass set with pooling installed to read all 1496 CLS vectors for the head fit. I'm wondering whether the pooling layer can be included in the experiment passes too, since reading the last kept layer would also give the final CLS, though that would hide the deeper layers from those reads.

I'm considering whether repurposing the last kept layer's attention heads for uniform mean pooling could replace a separate pooling layer. The idea is that the layer's FFN would then act on the pooled CLS vector, giving a nonlinear head on the mean of its input, which would avoid spending an extra layer on pooling.

Now I'm weighing a second pruning strategy: the pooling layer would be the last kept layer, so for M8 features I'd keep layers 0 through 7 plus pooling, and for M9 I'd keep one more. Its FFN only ever sees the CLS vector, so its m units can act as extra nonlinear kernel units alongside the pooler's 768 tanh units, and with m=3072 in the deep model there's plenty of room. For the real layers, I'm thinking about pruning FFN units by a weight-based score, such as the product of the norms of each unit's input and output weights, since I don't have activations to work from.

I'm checking whether any read-based method could reveal per-unit FFN activation statistics, but attention is needed to form the FFN input and no differential trick seems to work per unit. So I'm falling back on a weight-based magnitude baseline instead, since units with large negative biases tend to be mostly inactive under GELU.

I'm estimating which FFN units are rarely active by treating each unit's pre-activation as roughly Gaussian, with mean from the LN bias and variance from the gamma-scaled weights. The per-dimension statistics of the layer-normed input aren't directly readable, so this is only an approximation.

I'm estimating which FFN units are rarely active by treating each unit's pre-activation as roughly Gaussian, though BERT's outlier dimensions make that a crude proxy. The FFN input isn't directly readable from the layer outputs, so I'm considering a weight-based estimate instead.

I'm weighing which pruning amount to test next, since the savings per 512 units of m are substantial at this depth. I'm also reconsidering whether the approach stays within the rules, since computing the FFN activations directly from inputs I can only read indirectly isn't straightforward. Layer dropping in the middle of the stack is another option I want to keep in view.

I'm laying out a first experiment that keeps all twelve layers at m=1024 with the Gaussian-proxy pruning applied throughout, and reads mean_hidden at every layer over 1024 ids to get an accuracy profile by depth. I still need to make sure the reserved-channel heads in layers 2, 6, and 10 are handled by excluding those dimensions from the features.

I'm going to run the first experiment with all twelve layers kept at m=1024, using the Gaussian-proxy pruning throughout. Then I'll check whether a bias compensation for the dropped units helps or hurts, since it depends on how well the Gaussian assumption holds.

I'm working out how bias compensation for the dropped FFN units would look under a Gaussian model. Each unit's pre-activation would have mean from the LN beta and bias, and variance from the gamma-scaled weights, which gives a closed-form expected GELU output. I'm still unsure how well that assumption holds, since the real layer-normed inputs aren't white.

I'm estimating how much each FFN unit contributes by its output variance under a Gaussian approximation, scaled by the norm of its output weights. Units that rank low get dropped, and their mean contribution gets folded into the output bias. The white-noise assumption is shaky for BERT, since it has outlier dimensions with large gamma values, but it's a reasonable starting point.

I'm considering whether a data-driven read could get the FFN input, but recovering it would mean recomputing the attention and LayerNorm steps myself, which isn't allowed. So I'm sticking with the weight-based Gaussian proxy, and I'll check empirically whether pruning at that proxy holds up accuracy.

I'm weighing whether a pruned BERT can still carry the classifier's signal without data-driven reconstruction, since published FFN pruning methods lean on activation data I don't have. My guess is that refitting the classifier on the pruned features will absorb most of the drift, so the real question is whether class-relevant information survives. I'll run the first experiment and see.

I'm planning the pass structure for Exp 1, weighing m=768 against a 256 bracket depending on how it lands. Since CV on 1024 ids is noisy, I'll compare against a base run on the same subset, and I'm checking whether the first 1024 labels are balanced.

I'm settling on the pooling host now: it has to sit after the real layers, and since the kept layers share the same m and the pooling layer's FFN is off anyway, the choice doesn't change the output. I'll use layer 11 for it. The open question is whether to include that pooling layer in Exp 1 so I get the final CLS vector for fitting the head, or to hold off until the depth is decided.

I'm going to write the notes first and then compute the pruning scores. Layer 11 can host the pooling since its FFN is off anyway, and I'll fit the head on all 1496 ids in the final passes, with one predict pass to verify.

I'm pulling the FFN weights for each layer and checking how the editor zeroed out the reserved dimensions in layers 2 through 10, since those rows and columns should now contribute nothing. Next I'm building the Gaussian proxy scores: the pre-activation mean comes from the attention-output LayerNorm bias and the first-layer bias, and the variance from the squared weights scaled by the LayerNorm gamma.

I'm going to compute the Gauss-Hermite importance scores for every layer, then check how concentrated they are; if they're spread evenly, pruning will cost accuracy roughly in proportion to how many units I drop.



Shallow layers top out near 97.4%, so the model needs about 8–9 real layers, and I'll shrink the FFNs, which hold two-thirds of each layer's parameters. Without activations I'll rank FFN units by a weight-only Gaussian proxy, after saving notes.



## tool_use: run_python

```json
{
 "code": "notes_add = \"\"\"\n## Findings (base reads, pass_1/2 -> base_feats.npz; lib.py has folds/ridge/cv)\n- Linear ridge on mean_hidden per layer: L0 96.2, L1 96.0, L6 96.7, L9 97.2 (CV). [CLS] alone worse.\n- RBF kernel ridge (z-scored, /sqrt(d), gamma~0.4, lam~0.003) 3-seed CV: M0 97.1, M1 97.4, M6 97.6, M8 97.75, M9 98.2, M10 98.2,\n  M0+M8 98.3, M0+M9 98.5, additive K9+lin(M0) 98.5, K8+lin(M2,M6) 98.1. Shallow (k<=2) caps ~97.4 -> too risky.\n- Kernel can be realised in pooler: tanh(g*z.c + b) with b<<0 == exp-like == RBF on normalized inputs (identical acc). 768 centers lose ~0.3-0.5 vs full KRR.\n- Original model CV ~97.8 -> test 98.31. Target CV >= ~98.2.\n## Plan\n- Deep prefix (0..8) + pooling layer; shrink by FFN pruning (Gaussian proxy importance, bias compensation). Test m via reads of 1024 ids (all layers mean_hidden) -> CV per depth.\n\"\"\"\nwith open(WORK/'notes.md','a') as f: f.write(notes_add)\nimport numpy as np\nfrom math import sqrt\nxg, wg = np.polynomial.hermite_e.hermegauss(60); wg = wg/wg.sum()\nfrom scipy.special import erf\ndef gelu(u): return 0.5*u*(1+erf(u/np.sqrt(2)))\nstats={}\nfor l in range(12):\n    W1=base(f'bert.encoder.layer.{l}.intermediate.dense.weight'); b1=base(f'bert.encoder.layer.{l}.intermediate.dense.bias')\n    W2=base(f'bert.encoder.layer.{l}.output.dense.weight')\n    g=base(f'bert.encoder.layer.{l}.attention.output.LayerNorm.weight'); be=base(f'bert.encoder.layer.{l}.attention.output.LayerNorm.bias')\n    mu=W1@be+b1; sd=np.sqrt((W1**2)@(g**2))\n    U=mu[:,None]+sd[:,None]*xg[None]; G=gelu(U)\n    Em=(G*wg).sum(1); Ev=((G-Em[:,None])**2*wg).sum(1)\n    c2=np.linalg.norm(W2,axis=0)\n    imp=np.sqrt(Ev)*c2\n    stats[l]=dict(mean=Em,imp=imp)\n    s=np.sort(imp)[::-1]; cs=np.cumsum(s**2)/np.sum(s**2)\n    print(l, 'frac var in top 256/512/1024/1536:', [round(cs[k-1],3) for k in (256,512,1024,1536)], 'mu range',round(mu.min(),2),round(mu.max(),2),'sd med',round(np.median(sd),2))\nnp.savez(WORK/'ffn_proxy.npz', **{f'imp{l}':stats[l]['imp'] for l in range(12)}, **{f'mean{l}':stats[l]['mean'] for l in range(12)})\n"
}
```

