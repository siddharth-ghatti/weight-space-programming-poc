## reasoning summary

The m=2816 results look worse across the board: leave-one-out RBF drops 0.78 points, and the other variants also come out lower than base. m=2560 is not clearly better or worse, though; some of its numbers beat 2816 and even base on gelu20, so the noise here looks substantial.

The rbf5x5 drop is about 0.8 points, which fits the noise estimate of roughly ±0.3 to 0.5, so pruning costs somewhere around 0.2 to 0.8 overall. That is above my 0.15 threshold, so I'm reverting to m=3072. The revert means dropping the layer 0–8 bias edits and restoring the structure from backup. Then I'll confirm the only remaining edits are the layer 11 and pooler ones, and that the expected logits still hold for the m=3072 features. Pass 6 already checked 10 ids, and three passes remain for anything further.

I'm checking whether the m=3072 design still has accuracy headroom I can get offline, since the features come straight from the base reads. The RBF and GELU heads are close, and the main open option is a combined model that adds pooler exp-features, which seems like more complexity than it's worth. The 20-fold score of about 98.6 already looks solid against the original model's 97.8 on 5-fold and 98.31 on test.

I'm weighing whether a single-layer drop could save parameters, but dropping a whole layer looks likely to cost more accuracy than pruning the FFN units did, probably around 0.4 points, which is over my threshold.

I'm reverting to m=3072 and dropping the layer 0–8 bias edits, then doing a broader check of the logits across 1024 ids to catch any edge cases in the feature computation.



Pruning to m=2816 lost accuracy on every validation scheme, beyond my ≤0.15 threshold, so I'm reverting to the verified unpruned configuration and restoring the backup structure.



## tool_use: run_python

```json
{
 "code": "import shutil\nfor L in range(9): drop_edit(f'bert.encoder.layer.{L}.output.dense.bias')\nshutil.copy('structure_m3072_verified.json','structure.json')\ns=structure(); print(s['layers'], s['m'], param_count())\nprint(sorted(edits()))\ns=open('notes.md').read()\ns+='''\n## pass_7 result (m=2816): subset LOO rbf 96.97 vs base 97.75; gelu5x5 96.84 vs 97.11; gelu20 97.36 vs 97.56 -> REJECTED, REVERTED to m=3072 (structure_m3072_verified.json, L0..8 bias edits dropped).\n- Final candidate = verified m=3072 k=10 head (95,312,648 params). 3 passes left.\n'''\nopen('notes.md','w').write(s)\n"
}
```

