## reasoning summary

Stash signal at L9 is still exact, with the coefficient down to 0.036 and a small but reliable contribution. I'm checking whether the layer 9 readout, which pulls the stash dimensions through the B weights, is handled by the ridge regression on standardized features.

The layer-0 shrink of 0.45 looks like it comes from the embedding outputs and attention, which would explain the large sigmas there. Next I'm planning a full read of CLS and mean hidden states at layers 7, 8 and 9 for all 1496 samples, since layer 8 might give a better readout than layer 9.



The stash matches exactly, so the open question is whether the head-0 ablation hurt the CLS and mean features in layers 7–9. I'm checking those layers for ids 0–1023 next.



## tool_use: run_python

```json
{
 "code": "import json\nids=json.load(open('all_ids.json'))\njson.dump(ids[:1024],open('ids_a.json','w')); json.dump(ids[1024:],open('ids_b.json','w'))\nprint(json.dumps(ids[:1024]))"
}
```

