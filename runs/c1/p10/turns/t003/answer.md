## reasoning summary

Max length is 63 tokens, and practice records carry 'line', 'register', and 'variant' fields. Next I'll read the base model across all layers, pulling mean hidden, CLS hidden, pooled, and logits. The forward passes will run in two chunks, the first 1024 ids and then the remaining 472, so I'm jotting down notes before starting them.



Reserved dimensions and carry heads are identified; I'm reading the base model at every layer across all 1496 sentences next, in two passes because reads cap at 1024 ids.



## tool_use: run_python

```json
{
 "code": "notes = \"\"\"# Notes (shrinking insurance classifier)\n## Facts\n- reserved dims (carry channel): [83,88,222,254,444,495,506,601]; carry heads: L2 h0, L6 h11, L10 h10 (zero query)\n- classifier selects pooled dims 0-7 (identity), bias 0\n- practice: 1496 sents, max 63 tokens, median 29\n## Plan\n- pass1/2: read base all layers mean_hidden+cls_hidden+pooled+logits for all practice (ids[:1024], ids[1024:])\n- offline CV ridge per layer (EXCLUDE reserved dims: they hold leaked ridge scores) to find shallowest prefix with ~98%\n\"\"\"\n(WORK/'notes.md').write_text(notes)\nids = [p['id'] for p in P]\njson.dump(ids, open(WORK/'ids.json','w'))\nprint(len(ids))\n"
}
```

