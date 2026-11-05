## reasoning summary

Greedy selection gives the biggest gain on L4 (about 18% lower), with a small improvement on L1 (about 2%), which is close to what the g2sig variant achieved on L4. I'm considering combining the two: running greedy on a weighted covariance, weighting each sample by the inverse of its variance, and swapping in a different metric for W2.

Next I'll check the delta parameter and run the other variants across all layers. The v2 fit gave no change in end-to-end noise even though L3 and L4 improved, so the remaining error is probably coming from L0 to L2. To find out, I'm running two diagnostics: one with L0 and L1 pruned using v2 and the rest original, and one with L2 through L4 pruned and the rest original, then comparing both to the full v2 result.



Greedy selection helps: L4's holdout error drops 18% and L1's about 2%, but the L3/L4 gains left the final logits unchanged, so I'm running two diagnostic passes on the pruned layers next.



## tool_use: run_python

```json
{
 "code": "import os, shutil, json\nif not os.path.exists('edits_m1536_v2'): shutil.copytree('edits','edits_m1536_v2'); shutil.copy('structure.json','structure_m1536_v2.json')\n# D1: prune L0,L1 only (zeroing sim); restore L2-5 to original\nfor L in [2,3,4,5]:\n    for s in ['output.dense.weight','output.dense.bias']:\n        drop_edit(f'roberta.encoder.layer.{L}.{s}')\nprint(set_structure(layers=[0,1,2,3,4,5], ffn_keep=3072))\nprint(sorted(edits()) if not isinstance(edits(),dict) else sorted(edits().keys()))\n"
}
```

