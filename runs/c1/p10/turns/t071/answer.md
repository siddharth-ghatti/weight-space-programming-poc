## reasoning summary

The max error came out at 3.4e-6, so the harness matches the numpy expectation. The m=3072 design checks out on 10 samples, and since the formula is deterministic that should hold more generally, with the longest case at 63 tokens looking fine. With 4 passes left, the safe model is set at 95.3M, expected around 98.5, and I'm now deciding whether to spend the remaining passes on a smaller model.

I'm weighing whether to spend the remaining passes on a smaller m=2816 model. The pruning would be fairly cheap to verify, but the expected loss of about 0.3 is probably too large to accept under my criterion, so I'm inclined to skip it.

I'm weighing a smaller pruned model, but the expected loss is around 0.3 points, which likely fails my acceptance bar, so the odds of it being worth the remaining passes look low. Pruning by a Gaussian proxy looks weak, so a better importance measure would need actual token-level activation statistics from the base model.

I'm considering whether the layer input can stand in for the FFN input, since computing the attention output myself would mean reimplementing the forward pass, which I can't do. Using the layer output h_{L+1} as a proxy is the workable route, given the post-LN structure.

I'm looking at whether the token-level covariance of the layer input can stand in for the FFN input, and whether computing w_j^T Σ w_j from read-out hidden states counts as reimplementing the forward pass or just doing statistics on returned activations.

I'm weighing whether a better pruning criterion is worth it, since token-level covariance from returned hidden states is a statistical measure rather than a model forward pass. Even so, the importance spread looks flat, so perfect selection would still remove a few percent of energy at m=256.

I'm considering whether the pruned units' contribution could be approximated linearly and folded into a kept unit with a large positive bias, so that GELU acts nearly linear there, which would be a way to compensate for the removed capacity.

I'm considering whether a better importance measure could be built from token-level covariance of the hidden states, since that would beat the flat Gaussian proxy without reimplementing the forward pass. Unit merging looks harder to get right, and the linear-compensation route would need too many units to be worth it.

I'm checking whether the pruned-model plan can still fit in the remaining passes: an m=2816 experiment, then reading the rest of the tokens and one verification run, with the verified m=3072 config as the fallback if anything fails. The verification step only confirms the pooling formula I've already validated against numpy, so it adds little.

I'm weighing whether to skip the token-level proxy entirely and test m=2816 directly with the existing proxy, which would leave a spare pass. Either way, the pruned-FFN slicing for layer 11 is the only real risk, so verification is worth keeping if a pass is free.

I'm estimating the m=2816 loss from the pass-5 numbers: the m=2560 subset loss was about 0.48 (LOO) and 0.8 (5-fold), and if the damage scales with removed energy, m=2816 should land near 0.2 and 0.35. That puts the expected test score around 98.25, still clear of the 97.8 bar, though with a thinner margin than the m=3072 model's 0.7.

I'm weighing whether the remaining passes are better spent on a leaner model: an m=2816 test with the existing Gaussian proxy would be cheap, though a measured drop of 0.15 could still conceal a true loss near 0.3 given paired-sample noise. A token-level covariance proxy might select units better, but the flat importance spread limits how much that can recover at m=256.

I'm weighing whether a token-level covariance proxy would give a better importance estimate than the existing one, and whether that's worth the cost. The flat importance spread suggests I can't gain much at m=256 either way, so I'm leaning toward testing the pruned model directly and keeping the verified m=3072 config as the fallback.

I'm going to test the m=2816 pruning with the existing proxy, selecting the top 2816 units by importance and compensating with the proxy means, and I'll keep the layer 11 slice to units 0 through 1495 in mind.

Next I'm reading the edited model's layer-9 features for the first 1024 ids, to measure the drop against the base on the same subset. Then I'll check the compensation biases for layers 0 through 8 before running.

Next I'm setting the pruned layers to keep the top 2816 units by importance, with layer 11 keeping units 0 through 2815. I'll back up the current structure first so the verified m=3072 config can be restored exactly if the experiment is rejected.



The harness logits match my numpy expectation to 3.4e-6, so the m=3072 design is verified. I'll try m=2816 with four passes left, keeping it only if the loss stays at or below 0.15, and otherwise reverting to the verified setup.



## tool_use: run_python

```json
{
 "code": "import shutil, json\nshutil.copy('structure.json','structure_m3072_verified.json')\np=np.load('ffn_proxy.npz'); m=2816\nkeep={}\nfor L in range(9):\n    imp=p[f'imp{L}']; kp=np.sort(np.argsort(-imp)[:m]); pr=np.setdiff1d(np.arange(3072),kp)\n    W2=base(f'bert.encoder.layer.{L}.output.dense.weight'); b2=base(f'bert.encoder.layer.{L}.output.dense.bias')\n    save_edit(f'bert.encoder.layer.{L}.output.dense.bias', (b2+W2[:,pr]@p[f'mean{L}'][pr]).astype(np.float32))\n    keep[L]=kp.tolist()\nkeep[11]=list(range(m))\nprint(set_structure(layers=[0,1,2,3,4,5,6,7,8,11], ffn_keep=keep))\ns=open('notes.md').read()\ns+='''\n## pass_6: verified m=3072 head exact (logits err 3e-6 on 10 ids). Backup structure_m3072_verified.json.\n- EXPERIMENT m=2816 (proxy top-k + mean comp on 0..8; layer11 keeps 0..2815). pass_7 = read L9 ids p0000-p1023.\n  REVERT if loss > ~0.15: drop bias edits L0..8, copy structure_m3072_verified.json -> structure.json.\n'''\nopen('notes.md','w').write(s)\nprint(','.join(f'\"p{i:04d}\"' for i in range(3)))\n"
}
```

