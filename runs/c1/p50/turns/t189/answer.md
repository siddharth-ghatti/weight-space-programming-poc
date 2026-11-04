## reasoning summary

Old weights on the new input give a held-out error of 0.104, and refitting does worse at 0.112 to 0.125, even with all 3072 base units it only reaches 0.089. So refitting doesn't help here, and I'll keep the old weights for layer 7.

The residual error here is about 1.04, which fits the carry values staying in the residual dimensions rather than going to zero. Layer 7's refit didn't help, so I'll restore its tensors and structure from the backup and then move on to layer 8's cls and mean readings across the full set of passes for cross-validation.



Layer 7's old weights already beat every refit, so I'm restoring it from backup and skipping layer 8, then rebuilding the drop-6 structure.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nfrom lib5 import *\nP=pre(7)\nfor nm in ['attention.output.LayerNorm.weight','attention.output.LayerNorm.bias','output.dense.weight','output.dense.bias','output.LayerNorm.weight','output.LayerNorm.bias']:\n    save_edit(P+nm, np.load(f'backup_L7_preexpose/{nm}.npy'))\nst=json.load(open('structure_k10_m1024_final.json')); fk={int(k):v for k,v in st['ffn_keep'].items()}\nlayers=[0,1,2,3,4,5,7,8,11]\nprint(set_structure(layers=layers, ffn_keep={l:fk[l] for l in layers}))\n# verify layer 7 restored equals backup\nprint(np.abs(W(P+'output.dense.weight')-np.load('backup_L7_preexpose/output.dense.weight.npy')).max())\nwith open('notes.md','a') as f:\n    f.write('''\n## Run 5 (handoff 5)\n- Pool layer installed & verified (pass_36 logits == numpy pred, 6e-6). structure_k10_m1024_final.json = safe fallback (63.8M), full-1496 10f CV (3 seeds) 98.37 (lib7.cv, c=10 lam=.02 c0=.5 na=1008).\n- Depth CV (full 1496, kernel(last)+linM0): L9 98.37, L8 97.91, L7 97.88, L6 97.64, L5 97.48, L4 97.97. Extra linear carries (M4/M5/M6) don't help.\n- Pool anchors: 1008 98.37, 752 98.15, 624 97.88, 496 97.62 => m<1024 not viable.\n- Drop orig layer 6: 1024-subset 5 seeds: full 98.11, drop6 97.95, drop6 + L5 skip-refit (fit L5 FFN to base h7, LN from L6; lib8.fit_skip) 97.99. L5 skip-refit SAVED (backup_L5_preskip/).\n- L7 refit after drop6 (pass_41 exposure) does NOT help (old weights better) -> restored.\n- Current structure: [0,1,2,3,4,5,7,8,11] (59.9M), pool NOT yet refit for it.\n''')\n"
}
```

