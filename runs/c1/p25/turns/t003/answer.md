## reasoning summary

I'm settling on m=1 for each layer, since it adds almost no cost (1537 per layer), and I can neutralize a unit by zeroing its output column. That puts the minimum per layer at 2,367,745. Next I'll write up my notes, then run the base reads over all ids and layers, outputting both mean and CLS hidden states, in two passes.

I'm also going to check what mean_hidden actually means, so I'll look at the first pass's output before relying on it.



Each extra kept layer at m=1 adds about 2.37M params on top of the 24.43M floor, so I'm reading the base model's mean-pooled and [CLS] states across all 1,496 practice sentences to find how many original BERT layers I need to keep 98%. I'm writing notes first.



## tool_use: run_python

```json
{
 "code": "notes = \"\"\"# Notes (shrink insurance classifier)\n## Facts\n- m must be >=1 (m=1 costs 1537/layer). Per layer at m=1: 2,367,745. Floor 24,433,928.\n- Start model: carry heads in L2,L6,L10 (mean of their inputs -> 8 reserved dims), L11 = pooling layer, pooler rows 0-7 = ridge.\n## Strategy\n- last kept layer only matters at CLS -> use as pooling layer (pools its input). Question: how many BERT layers before it, what m.\n- Pass 1-2: base read mean_hidden+cls_hidden all layers, all 1496 ids -> CV ridge per layer.\n## Log\n\"\"\"\n(WORK/'notes.md').write_text(notes)\nids=[x['id'] for x in practice()]\njson.dump(ids, open(WORK/'ids.json','w'))\nprint(ids[:3], ids[1020:1026])\n"
}
```

