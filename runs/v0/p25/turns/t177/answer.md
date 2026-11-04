## reasoning summary

I'm checking the v5 results on these harder sentences, where it scored 98%. A few of the "errors" may just be my own debatable labels, like the pro-rata premium return and the short-rate refund, so those are ambiguous and I'll keep them. The "additional insureds" sentence might really be a definition or coverage item, so I'm considering dropping it.

I'm also weighing whether to drop the "Hospital" sentence, since its definitions label seems right to me. Next I'll set up the CV comparison, testing the v5 configuration against the version with my6 added at several weights.



The v5 model scores 98% (250/255) on the hard boundary sentences, though a couple of misses may be my labeling errors. Next I'll run cross-validation on practice data, with and without my6 in training, at several weights.



## tool_use: run_python

```json
{
 "code": "import sys, os; sys.path.insert(0, str(WORK)); os.chdir(WORK)\nimport numpy as np\nfrom kh import *\nD=load(); d5=np.load('my5_feats_clean.npz'); d6=np.load('my6_feats.npz')\nbase_extra=[(D['Zm'],D['ym']),(D['Z3'],D['y3']),(D['Z4'],D['y4']),(d5['Z'],d5['y'])]\nseeds=(0,1,2,3)\nprint('v5cfg', run_cv(D, base_extra, 0.4, 0.005, seeds))\nfor w6 in (0.4,0.7):\n    print('+my6',w6, run_cv(D, base_extra+[(d6['Z'],d6['y'])], [0.4]*4+[w6], 0.005, seeds))\n"
}
```

