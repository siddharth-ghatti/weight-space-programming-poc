## reasoning summary

I'm checking where things stand: the stash and L9 readout are saved and hit 100% on the practice set, though that's in-sample, so my honest estimate is around 98.0–98.1%. I have 11 forward passes and 230 code runs left, and I'm weighing whether to submit now or try a kernel readout, which looks only marginally better and carries some risk. I'm also thinking about whether I should check robustness issues first, such as long

I'm checking whether the tanh pooling saturates: if the scale is large, the outputs could bunch at ±1 and create argmax ties, so I want to see how big the logits are in the pass-14 predictions.



## message

I'll check the saved state first: the logit magnitudes from pass 14, so I can rule out tanh saturation and argmax ties, and the files in the working folder.

## tool_use: run_python

```json
{
 "code": "\nimport json, numpy as np, os\np=json.load(open('pass_14/predictions.json'))\nitems=p['items'] if isinstance(p,dict) and 'items' in p else p\nprint(type(p), (list(p.keys())[:5] if isinstance(p,dict) else len(p)))\n"
}
```

