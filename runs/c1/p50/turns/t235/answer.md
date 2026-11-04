## reasoning summary

The skip-err number is a poor predictor here: drop2 had a lower skip error than drop4 but still cost 0.4, while drop4 cost about nothing, so the outcome depends on which features survive rather than on per-token reconstruction error. That means I can't trust it as a shortcut, and each further drop experiment needs at least two passes, an eval read followed by a 472 read and verification. I'm now weighing a three-pass plan, starting with the eval read at 1024 and moving on to the 472 read only if that looks good.

I'm weighing whether dropping layer 1 is the only workable candidate, since dropping layer 3 would leave the skip layer needing to map across three layers, which doesn't hold. Dropping layer 1 means layer 0's output gets its residual gamma set to 1 to carry the signal forward, and the layer 2 residual reads stay cut as before. The biggest relative gain is in that drop1 skip-refit, but the drop2 result, down 0.4 on the subset, makes me wary of trusting it.

I'm stopping the drop experiments here: the estimate of 98.25 against a 97.8 threshold leaves too little room, since another 0.2 to 0.4 loss would bring it near 97.9 to 98.05. What's left is to check that structure.json is valid and the notes are current, confirm the classifier and pooler edits are intact, and then submit.



The screen didn't help: drop2 scored better on reconstruction error than drop4 but cost 0.4% on classification, so per-token error isn't a reliable guide. I'm stopping here and submitting the verified model, with a last check that the structure file is valid.



## tool_use: run_python

```json
{
 "code": "import json\nst=structure(); print(st['layers'], st['m'], {k:len(v) for k,v in st['ffn_keep'].items()} if isinstance(st['ffn_keep'],dict) else 'list')\nprint(param_count())\nwith open('notes.md','a') as fh: fh.write(\"- Skip-refit token err does NOT predict drop cost (drop2 skip err .111 < drop4 .141 but cost .4 vs 0). Further drops not attempted. SUBMITTING [0,1,2,3,5,7,8,11] m=1024, 55,954,696 params.\\n\")"
}
```

