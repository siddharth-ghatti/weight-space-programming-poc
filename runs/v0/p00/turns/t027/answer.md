## reasoning summary

Zeroing barely changes anything, with the score moving from 97.24 to 97.30, so the per-class F1 is balanced. I'm weighing whether to push for more, but the remaining options are either marginal or not worth the risk, and the table is already in good shape. I'm close to finished, so I'll update the notes and submit, though I want to double-check one risk first: how the harness might compute its score.

I'm checking whether the harness treats the edited model as the base plus additive deltas or as replacement. The save_edit and W docs both point to replacement: W returns the saved array as the current tensor, so the saved edits stand in for the base tensors. The classifier starts at zero, so it behaves the same either way, but the other tensors would differ. Next I'll confirm the edit count and that everything is finite, then update the notes and submit.



Zeroing the special tokens barely changes CV accuracy (97.2% vs 97.3%), and per-class F1 is balanced. I'm running a final integrity check on the saved edits and notes.



## tool_use: run_python

```json
{
 "code": "cat={c['name']:c for c in catalog()}\nbad=[]\nfor e in edits():\n    a=W(e)\n    if not np.all(np.isfinite(a)) or tuple(a.shape)!=tuple(cat[e]['shape']) or not cat[e]['editable']: bad.append(e)\nprint('edits', len(edits()), 'bad', bad)\nnotes=open('notes.md').read()\nnotes+=\"\"\"\n## Final checks (run 27)\n- CV with CLS/SEP scores zeroed (as built): 97.2% acc, per-class F1 0.957-0.982.\n- Bigram restricted-set gain only +0.3-0.7% CV -> skipped (prev-token head too risky with 0 forward passes).\n- Detectors robust to positions up to 127; specials ->0 at all positions up to 511; unseen tokens ~0 output.\n- 108 edits all finite/correct shapes. READY TO SUBMIT v1.\n\"\"\"\nopen('notes.md','w').write(notes); print('notes updated')"
}
```

