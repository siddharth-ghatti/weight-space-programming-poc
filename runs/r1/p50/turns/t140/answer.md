## reasoning summary

The noise levels are heavy-tailed: the median is about 0.09, but the 99th percentile sits near 0.47, roughly five times higher. Noise across the two embedding sizes correlates at 0.75, so the same items tend to be noisy in both. Items p0557 and p0456 look like near duplicates from UBS CDP, and I'm wondering whether some items are sensitive because they sit near a decision boundary in a high-curvature part of the representation.

I'm weighing whether the noise is simply a property of the inputs, since compensation can't really fix it. Sensitive items have a larger logit-noise term, and making the model itself less sensitive doesn't look workable. So the real choice is among m1536, m1792, and m2048. I'll build m1792 and run one evaluation to fill in the curve before deciding, and I want to check whether the items used to fit compensation show lower noise than the rest, which would make the practice estimates a bit optimistic.



Item noise is mostly driven by the inputs, correlating 0.75 across m1536 and m2048, so compensation can't fix it. I'm building m1792 next to complete the size–risk curve and checking how optimistic in-sample noise is for the fitting ids.



## tool_use: run_python

```json
{
 "code": "import shutil, json, os\nexec(open(WORK/'fit2.py').read())\npc=build2(1792, lam=0.1, keepfile='keep_m1792_v2.json')\nprint('params',pc)\nshutil.copytree('edits','edits_m1792_v2', dirs_exist_ok=True); shutil.copy('structure.json','structure_m1792_v2.json')\n# in-sample check\nids=json.load(open('all_ids.json')); bl=np.load('base_logits.npy')\nfit_ids=set(json.load(open('probe_ids.json')))\nb=json.load(open('batches.json')); print(type(b), list(b.keys()) if isinstance(b,dict) else len(b))\nfor v in (b.values() if isinstance(b,dict) else b): fit_ids|=set(v)\nlj=json.load(open('lens.json')); lm=dict(zip(lj['ids'],lj['lens'])); Ln=np.array([lm[i] for i in ids])\nisfit=np.array([i in fit_ids for i in ids]); print('n fit ids',isfit.sum())\nfor f in ['lg_pass25.npy','lg_pass30_m2048v2.npy']:\n    lg=np.load(f); z=np.sqrt(((lg-bl)**2).mean(1))*np.sqrt(Ln/100)\n    lo=(Ln>=120)\n    print(f,'fit ids z rms %.3f  nonfit (len>=120) %.3f  nonfit all %.3f'%(np.sqrt((z[isfit]**2).mean()), np.sqrt((z[~isfit&lo]**2).mean()), np.sqrt((z[~isfit]**2).mean())))\n"
}
```

