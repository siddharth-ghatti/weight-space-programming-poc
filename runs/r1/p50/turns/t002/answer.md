## reasoning summary

I'm starting fresh by sketching out a plan, and then the first pass will be a base read over all 1000 ids, capturing the CLS and mean hidden states, pooled output, logits, and every layer.



I'm running one pass of the base model over all 1000 practice paragraphs to capture its predictions and per-layer representations, so I can judge offline how many layers can be dropped before I edit anything.



## tool_use: run_python

```json
{
 "code": "notes = \"\"\"# Notes (climate TCFD pruning)\n## Strategy\n- Pass: base read all 1000 ids (cls/mean hidden all layers, pooled, logits) -> offline closed-form probes per layer.\n- Then: drop layers, refit head (ridge / lstsq, closed form) on edited model's <s>, narrow FFN.\n- Target fidelity to base predictions (test acc 77.5, need >=76.5, F1>=0.6896); leave margin.\n## Log\n\"\"\"\n(WORK/'notes.md').write_text(notes)\nids = [x['id'] for x in practice()]\njson.dump(ids, open(WORK/'all_ids.json','w'))\nprint(len(ids))"
}
```

## tool_use: forward_pass

```json
{
 "kind": "read",
 "model": "base",
 "ids": [
  "p0000"
 ],
 "outputs": [
  "logits"
 ]
}
```

