## reasoning summary

I'm weighing the honest filter's results: a small gain of about 0.2%, with most paired seeds improving, though a couple got worse.

Now I'm checking the disagreeing sentences to see whether my labels or the model are off. The idea is that if the labels are wrong I should fix them, but if the model is wrong those sentences still help it learn new regions. Cross-validation suggests adding all of them doesn't help on the practice data, which may mean the practice labeling convention differs from mine on those cases.



With the per-fold filtering applied, the gain is only about +0.2% (0.9801 to 0.9821, better on 6 of 8 splits), which is modest and noisy. Next I'll examine the roughly 45 disagreeing sentences to see whether my labels or the model are at fault.



## tool_use: run_python

```json
{
 "code": "import json\nA=json.load(open('my_sents_all.json')); ym=np.array(A['labels']); pm=np.load('my_pred_by_practice.npy')\nfor i in np.where(pm!=ym)[0]:\n    print(i, f\"{LABELS[ym[i]][:10]:10s}->{LABELS[pm[i]][:10]:10s} | {A['texts'][i][:140]}\")\n"
}
```

