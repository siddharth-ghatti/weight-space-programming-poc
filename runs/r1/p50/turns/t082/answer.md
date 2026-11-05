## reasoning summary

The notes were never updated after the sequential collection, which used passes 13 through 24 for the L0–L4 B1/B2 probes and the L5 classifier. Pass 24 seems to be the cls_hidden_L6 run, and pass 25's predict at m1536 came out at .91 accuracy. Next I want to compare pass_25's predictions against the base predictions and go through the metadata for passes 13–24, along with seq.py, keep_sets.json, and builder.py.



Notes stop at the plan, but passes 13–24 seem to show the sequential collection ran, so I'm checking the pass metadata and seq.py to reconstruct it.



## tool_use: run_python

```json
{
 "code": "import json, numpy as np, os\nfor k in range(12,26):\n    d=f'pass_{k}'\n    if os.path.exists(d+'/meta.json'):\n        m=json.load(open(d+'/meta.json'))\n        print(k, {kk:(v if not isinstance(v,list) or len(v)<10 else f'list{len(v)}') for kk,v in m.items()})\n    else:\n        print(k, os.listdir(d))\nprint(open('seq.py').read())\n"
}
```

