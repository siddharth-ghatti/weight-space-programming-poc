## reasoning summary

I'm weighing whether to run a single probe pass over the 32 longest sequences, which would add roughly 10k tokens and bring the total to around 14k, mainly to help the early layers.

I'm checking whether the layer-0 error improvement from more data is worth the extra cost, and it looks like roughly 25% less error. For layers 0 through 2 I'd run base-model probes on the 32 longest sequences, which avoids sequential dependencies.

I'm settling on the plan to run three base-model probes for layers 0 through 2 on the longest sequences, then rebuild the layer-0-to-2 estimates from roughly 14k tokens. After that I'll check whether a larger probe budget or a learning-curve fit is worth the extra passes.

I'm settling on which 32 longest sequences to use for the probe passes, and I'll drop the existing edits first so each layer is probed against the base weights.

I'm checking the candidate probe ids against the ones already in use, so I can pick the 32 longest that don't overlap.



The 32 longest paragraphs would cost about 10,300 tokens per probe pass, versus 4,051 now, so I'll pick the 32 longest unprobed ids and set up the layer-0 probe on the base weights next.



## tool_use: run_python

```json
{
 "code": "import numpy as np, json\nexec(open('probe.py').read())\nL=np.load('tok_len.npy')\nprobe_ids=set(json.load(open('pass_5/meta.json'))['ids'])\no=np.argsort(-L,kind='stable')\nnew=[f'p{i:04d}' for i in o if f'p{i:04d}' not in probe_ids][:32]\nprint(new, sum(L[[int(i[1:]) for i in new]]))\njson.dump(new,open('probe2_ids.json','w'))\nprint(setup_probe(0, save_edit, drop_edit, edits, set_structure))\nprint(edits())\ns=open('notes.md').read()\ns+='''- pass13 RESULT (m=1536 L0-4 + L5 probe): upstream R2 roughly 2x worse than m=2048 (L5 cls .918 vs .959; mean .935 vs .970). L5 refit: local .949, corrected-target .9556 = same as unpruned FFN on edited input (.9558) vs m2048 final .979. => sequential correction weak; upstream L0-2 error dominates.\n- Plan: base probes with 32 LONGEST ids (probe2_ids.json, ~10.3k tokens) for L0,L1,L2 (passes 14-16) -> more LS data. Edits DROPPED during probes; rebuild with lsbuild afterwards.\n'''\nopen('notes.md','w').write(s)\n"
}
```

