## reasoning summary

Layer 9, right after layer 8, looks like the sweet spot at roughly 98.7% LOO, with layer 8 alone trailing at 97.8%. The current setup is layers 0 through 8 plus pooling, with k=10 and about 95.3M parameters. With six passes left, I'm weighing the mild-pruning experiment against something that might have more upside, and I'm wondering whether building around layer 8 could help, given how sharp that jump is.

I'm weighing whether dropping an earlier layer, like 1, 2, or 3, could work while keeping layer 8. A single drop would probably cost around 0.3 to 0.5 points, which is a real risk to the margin, and I'd need to see almost no degradation before adopting it, something I'd put at roughly a 25% chance. The mild-pruning option could save more parameters but the crude proxy makes it hard to predict, so I'd estimate about a 40% chance it holds up.

I'm settling on a pooling layer at layer 11 so the final read can reuse the experiment's 1024 ids and the head needs only one more read of 472. Then I'll run a third experiment on the layer 0-8 structure with 2560 units kept per layer, reading mean and CLS hidden states, and compare the layer 9 LOO against the 97.75% baseline.

Next I'm weighing whether to run the fallback base config first, which would mean reading the final CLS features for the layer 0-8 structure before the experiments. That keeps the safe option secured, and then the remaining passes can go toward the riskier pruning ideas.

I'm going to lock in a safe solution before spending more passes, since the layer 0-8 structure with pooling is already well characterized. Computing the final features from the base reads looks possible, because the pooling step is just an average followed by layer norm, though reading them directly would be more reliable.

I'm weighing whether to fit the pooler on values I compute in numpy from the known base reads, then check them with a predict pass. The pooling step is an average plus residual and LayerNorm, but I'd rather read the actual final CLS features from the edited model, so I'll do that for the verification.

Next I'm laying out the remaining passes: install the pooling layer on the 0-8 structure at 3072 units, read the first 1024 items' outputs, then the rest, fit the head, and finally run a 512-item predict pass to check the logits against numpy.

I'm weighing whether to explore a more aggressive pruning config first, since that keeps the safe fallback available: installing the pooling layer at 2560 units would cost one pass, and if it holds up I could still afford a further reduction. If it doesn't, I'd fall back to the 3072-unit version with two reads and a verification pass.

I'm weighing an experiment at 2560 units per layer against the sharper pruning options, and I'm checking how much proxy variance each layer would lose at that width, which I expect to be only a small share.

I'm weighing the 2560-unit mild pruning first, since the experiment's expected payoff is worth one pass, with m=2048 as a follow-up if that holds up. I'll also include compensation in the test.

I'm going to include compensation for the pruned units, though their mean contribution should be small since the dropped units have little output weight.

Next I'm setting up the layer-11 pooling block: zeroing the query so attention is uniform, setting the value projection to identity, and scaling the output projection to add the mean of the hidden states back onto the CLS token.

I'm setting the layer-11 block so its attention averages the hidden states and its feed-forward path outputs only the bias, with the reserved dimensions zeroed out. The final CLS vector then comes from a layer norm over that result, and I'll fit the pooler on the values I read back from the edited model.

I'm settling the pooling layer's details: the harness pads batches but masks the pads, so attention is uniform over real tokens and the mean hidden state is over those tokens too. For the sub-2560 experiment I'll prune layers 0 through 8 and keep the first 2560 units in layer 11. The reserved dimensions at layer 9 still carry leaky scores, but zeroing the output rows and the LayerNorm gain should keep them out of the pooled vector.

I'm weighing whether to zero the carry heads' output in layers 2 and 6 so the leaky reserved scores stay out of the pooled vector entirely. The leakage through the layer norm statistics looks negligible, since the squared score norm is roughly class-independent, so I'm inclined to leave the heads as they are.

I'm weighing whether to zero the carry-head outputs in layers 2 through 10 so the reserved dimensions stay clean. That would make the no-leakage story tidier, but it would also move the network away from the base reads, so I'd be relying on the final read to catch any difference.

I'm leaving the carry heads untouched, since the leakage through the layer norm is negligible and the reserved dimensions shouldn't count as class evidence in my head anyway. Next I'll write the Exp 3 edits: the pooling layer on top of proxy pruning to m=2560 with compensation for layers 0 through 8, and a note to record the setup.



Layer 9 gives the best leave-one-out accuracy (about 98.7%), so the design is fixed at layers 0–8 plus one pooling layer. I'm now testing mild FFN pruning with 6 passes left, and installing the pooling layer in the same pass.



## tool_use: run_python

```json
{
 "code": "import numpy as np, json\nRES=np.array([83,88,222,254,444,495,506,601]); ALPHA=5.0\ndef install_pool(host=11, alpha=ALPHA):\n    p=f'bert.encoder.layer.{host}.'\n    save_edit(p+'attention.self.query.weight', np.zeros((768,768),np.float32))\n    save_edit(p+'attention.self.query.bias', np.zeros(768,np.float32))\n    save_edit(p+'attention.self.value.weight', np.eye(768,dtype=np.float32))\n    save_edit(p+'attention.self.value.bias', np.zeros(768,np.float32))\n    O=alpha*np.eye(768,dtype=np.float32); O[RES,:]=0\n    save_edit(p+'attention.output.dense.weight', O)\n    save_edit(p+'attention.output.dense.bias', np.zeros(768,np.float32))\n    g=np.ones(768,np.float32); g[RES]=0\n    save_edit(p+'attention.output.LayerNorm.weight', g); save_edit(p+'attention.output.LayerNorm.bias', np.zeros(768,np.float32))\n    save_edit(p+'intermediate.dense.weight', np.zeros((3072,768),np.float32)); save_edit(p+'intermediate.dense.bias', np.zeros(3072,np.float32))\n    save_edit(p+'output.dense.weight', np.zeros((768,3072),np.float32)); save_edit(p+'output.dense.bias', np.zeros(768,np.float32))\n    save_edit(p+'output.LayerNorm.weight', g); save_edit(p+'output.LayerNorm.bias', np.zeros(768,np.float32))\ninstall_pool()\nZ=np.load(WORK/'ffn_proxy.npz')\ndef prune(m, layers):\n    keep={}\n    for l in layers:\n        imp=Z[f'imp{l}']; mean=Z[f'mean{l}']; order=np.argsort(-imp); drop=order[m:]\n        keep[l]=[int(i) for i in np.sort(order[:m])]\n        W2=base(f'bert.encoder.layer.{l}.output.dense.weight'); b2=base(f'bert.encoder.layer.{l}.output.dense.bias')\n        comp=W2[:,drop]@mean[drop]\n        if l>=2: comp[RES]=0\n        save_edit(f'bert.encoder.layer.{l}.output.dense.bias',(b2+comp).astype(np.float32))\n        s=np.sort(imp)[::-1]; fr=(s[:m]**2).sum()/(s**2).sum()\n        print(l,'kept proxy var frac',round(float(fr),3),'comp norm',round(float(np.linalg.norm(comp)),3))\n    return keep\nkeep=prune(2560, range(9)); keep[11]=list(range(2560))\nprint(set_structure(layers=[0,1,2,3,4,5,6,7,8,11], ffn_keep=keep))\nwith open(WORK/'notes.md','a') as f: f.write(\"\"\"\n## Exp2 result (pass_4, drop 5&7): after-8 subset LOO 96.97 vs base M9 97.75 (subset); adding 9,10 back only recovers to 97.66. Layer dropping not worth it.\n- Full-data LOO (alpha=5 mix LNr(5M+C)): L7 97.8, L8 97.8, L9 98.7, L10 98.5, L11 98.4 -> design = layers 0..8 + pooling layer (host 11), k=10.\n- Pooling layer installed (install_pool in run ~35): q=0, V=I, O=5*I (res rows 0), LN g=1 b=0 (res 0), FFN zero.\n- Exp3 (pass_5): m=2560 proxy pruning + compensation on 0..8, structure [0..8,11]; read 1024 ids. Compare idx9 subset LOO vs base 97.75. Fallback: m=3072 (95.3M) -> drop pruning bias edits for 0..8!\n\"\"\")\n"
}
```

