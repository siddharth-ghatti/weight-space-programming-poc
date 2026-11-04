## reasoning summary

I'm starting fresh by writing up the notes and plan, and then I'll run the two read passes for mean_hidden and cls_hidden across all layers.



Next I'll read activations for all 1496 practice sentences, probe each layer offline, and then rewire the network so its final [CLS] vector carries the best-performing features.



## tool_use: run_python

```json
{
 "code": "notes = \"\"\"# notes.md\n## Plan (run 1)\n- Read mean_hidden + cls_hidden all layers for all 1496 practice (2 read passes).\n- Probe offline with closed-form ridge/LDA + CV to pick best layer.\n- Edit: layer L+1 aggregates (query=0 -> uniform attention, V=I, attn-out dense=c*I) so CLS = LN(mean tokens);\n  later layers -> identity (attn out dense=0, ffn out dense=0, LN gamma=1 beta=0).\n- Then read edited model's real CLS/pooled, fit closed-form head (pooler rows = s*A, classifier = identity selection).\n## Log\n\"\"\"\nopen(WORK/'notes.md','w').write(notes)\nids = [p['id'] for p in P]\njson.dump(ids, open(WORK/'ids.json','w'))\nprint(ids[:3], ids[1020:1026])\n"
}
```

