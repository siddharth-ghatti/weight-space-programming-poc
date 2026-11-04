## reasoning summary

I'm noting that the layer 0 key weights were saved unchanged, which is fine since the edit just rewrites identical values. Next I'll update the notes and set up the read pass over the first 1024 ids, collecting the cls hidden states for layers 1 through 9 and the mean hidden state at layer 9, then check whether the layer 0 mean state matches the original since the embeddings should be frozen.

I'm going to check the format of all_ids.json while I update the notes.



The stash edit is done across 176 tensors, with the layer-0 Wo stash block verified as exactly 4·I. Next I'll update the notes and do a full read of the first 1024 practice ids for CLS states and mean_hidden.



## tool_use: run_python

```json
{
 "code": "import json\nids=json.load(open('all_ids.json')); print(type(ids), len(ids), ids[:3])\nopen('notes.md','a').write('''\n## Session 2\n- Repeated-CV (3 seeds, folds_rep.npy): linear stdz[C9,M9] 97.64; +M0 98.08; kernel(GELU cos) on stdz concat ~98.0; kernel+M0 stash ~98.17.\n  Extra mean layers beyond M0 add nothing. Realizable kernel inside layer 9 (M9 PCs in C9 low-var dirs, LN) only ~97.8 -> deprioritized.\n- STASH design (stash.py, stash_cfg.npz): layer0 head0 Q=0 (uniform; head0 was near-uniform TV .185), value dims J -> A0 (M0-part of joint ridge\n  lam1000, zero-mean, std~.1-.16), Wo writes c_s=4 into stash dims S=[654,529,216,175,145,159,531,517]; layers0-8: reads of S mean-ablated\n  (Q/K/V via M_l mean, W1 via M_{l+1} mean), writes to S zeroed, LN gamma_S=median gamma, beta_S=0. Readout L9 must be refit on C9',M9'.\n- Saved stash edits (layers 0-8) on top of L9 readout. Next: read C1..C9', M9' for all ids (2 passes).\n''')"
}
```

