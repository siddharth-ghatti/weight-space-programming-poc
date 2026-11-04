## reasoning summary

The carry checks out cleanly: the R coordinates come out as the scaled signal plus a shared shift, with zero residual and full argmax agreement. The scale wobbles by about ±10% across the final sentences, which is fine, and with γ=1 it grows gently from 1 to roughly 1.7–2.2, still acceptable. I'm now looking at drift in the non-R coordinates, around 5–10% in relative norm, and I'm trying to pin down how much the ablation of eight coordinates and one head actually moves the downstream representations.

I'm checking how the ablation shifts the layer 3 representations, and the drop there looks like it's mostly from the sacrificed head. Next I want to see whether the m11 quality holds up. The layer 12 scale of 4.6 seems to come from layer norm standardization, so that's expected. I'll pull the final CLS features for the remaining 1240 examples, which means two more read passes, and also grab layers 11 and 12 for the mean and CLS hidden states.



The carry holds at every layer after 2 with zero residual and full argmax agreement, and the scale drifts gently with about ±10% variation between sentences. I'm reading the final [CLS] and L11 features for the remaining 1240 sentences next, so I can fit the head on real edited-model activations.



## tool_use: run_python

```json
{
 "code": "ids=json.load(open(WORK/'ids.json')) if False else None\nimport json\nids=json.load(open(WORK/'ids.json'))\nprint(json.dumps(ids[256:1280]))\n"
}
```

