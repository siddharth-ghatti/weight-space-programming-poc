## reasoning summary

Next I'm checking whether dropping some of the layers 0 through 8 still keeps accuracy near 98.0%. The skip experiment suggested layer 4 might be removable at little cost, so I'm planning to test that with passes.

I'm weighing whether dropping layer 4 keeps accuracy near 98.0%, testing the layer set [0,1,2,3,5,6,7,8,9,10] and comparing its edited depths against the base layers. If that holds, I could try dropping a second layer too, such as 3 or 6.

I'm also weighing whether layer 0 is essential. Dropping layer 8 and keeping layer 9 instead is a possibility too, but the skip experiment already showed that losing layer 8 costs more, so I'm less inclined to try that.



The z-score-only kernel matches the L2-normalised one, so I'm targeting layers 0–8 plus a pool slot (95.3M) unless a layer can be dropped for free. Next I'm testing the drop-4 pattern against base L9.



## tool_use: run_python

```json
{
 "code": "for e in list(edits()): drop_edit(e)\nprint(edits(), set_structure(layers=[0,1,2,3,5,6,7,8,9,10], ffn_keep=3072))\nwith open(WORK/'notes.md','a') as f:\n    f.write(\"\"\"- Exp D result: m=1536 -> L9+linM1 97.82->97.07 (first1024). FFN pruning not viable for deep path.\n- Realistic readout CV (full set, honest): kernel(z-scored M9, no L2) + lin(M2) ~98.0; M8 ~97.55. Target structure: layers 0..8 + pool slot (k=10, 95.3M) unless a layer can be dropped free.\n- Exp E (pass8): drop layer 4 only: [0,1,2,3,5,6,7,8,9,10], read means 1024. (all edits dropped before)\n\"\"\")\n"
}
```

