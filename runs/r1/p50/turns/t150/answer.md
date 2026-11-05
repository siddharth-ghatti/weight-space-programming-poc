## tool_use: run_python

```json
{
 "code": "import json\nbl=np.load('base_logits.npy'); ids=json.load(open('all_ids.json'))\npr=json.load(open('pass_32/predictions.json')); idx={r['id']:i for i,r in enumerate(pr)}\nlg=np.array([pr[idx[i]]['logits_edited'] for i in ids])\nl26=np.load('lg_pass26.npy')\nprint('max |diff| vs pass26', np.abs(lg-l26).max(), 'agr w base', (lg.argmax(1)==bl.argmax(1)).mean())\nwith open('notes.md','a') as f:\n    f.write(\"\"\"\n## Session 4 (from turn 120)\n- Logit-error systematic part tiny (CV 4x4 logit correction: rms .170->.164, no agr gain).\n- Noise ~1/sqrt(len); item sensitivity strongly tied to low base margin and aligned with decision direction (Gaussian iid model underestimates flips).\n- No in-sample advantage for fit ids (noise generalizes).\n- pass30: m2048 v2 (build2) agr .984 rms .106. pass31: m1792 v2 agr .980 rms .139.\n- risk.py + donor sim (noise from margin-matched donors in rank frame, test logits = base/k): P(fail) at k=1.25:\n  m1536 v1 .164, v2 .136, v3 .133; m1792 ~.11-.13; m2048 ~.07. Linear EV favors m1536 (m1280 extrapolated ~equal EV, higher var).\n- FINAL CHOICE: m1536 v2 restored (edits_m1536_v2 + structure_m1536_v2.json), pass32 confirms identical to pass26 (agr .980). Submit.\n\"\"\")\n"
}
```

