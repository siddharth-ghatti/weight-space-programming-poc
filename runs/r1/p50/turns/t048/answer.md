## reasoning summary

The one-shot OBS result barely beats the var-rank+ridge baseline on holdout, and the train-test gap is wider, so the data is the real bottleneck. I'm also wondering whether a different error weighting would matter more, though that starts to feel too complicated. Maybe I should look at a fundamentally different way to cut the parameter count.

I'm weighing whether to cut layers, for instance by dropping layer 4, but that costs agreement (.955), worse than m1536. Layer 5 only needs the CLS row, though its FFN width has to match the others anyway. With the width shared, the question is how much total noise the budget can absorb, and whether m1024 might reach around .97 with four times the data. I'm also considering an affine head refit on the final CLS, which might let me reduce the dimensionality further.

I'm going to test whether refitting the head on the edited CLS features improves cross-validated agreement over the current .984. I'll try two options: a ridge correction of the logit difference, and a linear map from edited CLS back to base CLS before the original head.



Selecting features with OBS doesn't beat var-rank+ridge on held-out data and overfits more, so data volume is the real limit. I'm next testing a cheap cross-validated head correction on pass 12 before collecting more data.



## tool_use: run_python

```json
{
 "code": "import numpy as np, os\nexec(open('common.py').read())\nprint(sorted(os.listdir('pass_2'))[:30])\nC_ed = np.load('pass_12/cls_hidden_L6.npy').astype(np.float64); lg_ed = np.load('pass_12/logits.npy')\nC_b = np.load('pass_2/cls_hidden_L6.npy').astype(np.float64)\nWd = base('classifier.dense.weight'); bd = base('classifier.dense.bias'); Wo = base('classifier.out_proj.weight'); bo = base('classifier.out_proj.bias')\nhead = lambda C: np.tanh(C@Wd.T+bd)@Wo.T+bo\nreport('ed head', head(C_ed)); report('base recompute', head(C_b))\n# (b) map C_ed -> C_b residual via ridge CV\nfor lam in [10,100,1000,10000]:\n    D = ridge_cv(C_ed, C_b - C_ed, lam, folds)\n    report(f'map lam{lam}', head(C_ed + D))\n# (a) logit correction\nfor lam in [100,1000,10000]:\n    D = ridge_cv(C_ed, BL - lg_ed, lam, folds)\n    report(f'logit corr lam{lam}', lg_ed + D)"
}
```

