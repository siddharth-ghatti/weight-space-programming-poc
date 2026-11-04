## reasoning summary

I'm checking how the labels are stored: they turn out to be strings, which is why the examples didn't print as integers. Next I'll write up notes.md and then look at the examples.



## message

The labels are strings, so I'll map them to indices. Next I'll write the plan to notes.md and look at some examples.

## tool_use: run_python

```json
{
 "code": "notes = \"\"\"# notes.md\n## Constraints\n- 0 forward passes. No simulation of model. Must design weights analytically.\n## Plan (v1)\n- Layer0: attention no-op (attn.output.dense W,b = 0). attn LN gamma=1,beta=0 except 8 feature dims D_f with gamma=0 -> h0[D_f]=0.\n- Layer0 FFN: keyword detector neurons (one per informative wordpiece token), on h0=standardize(LN_emb(E+P+T)).\n  output dense writes per-token class score vector (centered across classes) into D_f.\n- Layer1: attention Q=K=0 -> uniform mean over tokens; V selects D_f; Wo writes mean scores into D_f of CLS.\n- Layers 2-11: no-op (attn out dense=0, ffn out dense=0, LN gamma=1 beta=0).\n- Pooler: pick D_f dims (moderate scale, tanh monotone); classifier = identity on 8 dims, bias 0.\n- Token scores: closed-form NB / ridge on bag-of-wordpieces from practice data (eval by CV on data only).\n\"\"\"\nopen('notes.md','w').write(notes)\nP = practice()\nlab2i = {l:i for i,l in enumerate(LABELS)}\nprint(LABELS)\nimport collections\nprint(collections.Counter(p['register'] for p in P))\nprint(collections.Counter(p['line'] for p in P))\nprint(collections.Counter(p.get('variant') for p in P))\nimport random\nrandom.seed(1)\nfor lab in LABELS:\n    ex = [p for p in P if p['label']==lab]\n    random.shuffle(ex)\n    print('=== ', lab)\n    for e in ex[:6]: print('   ', e['text'])"
}
```

