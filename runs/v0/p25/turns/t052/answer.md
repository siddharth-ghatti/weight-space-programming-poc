## reasoning summary

The learning curve is still climbing, from 897 to 1197 samples gives about +1.0%. Extrapolating to all 1496 suggests roughly 0.984, and another ~1000 good sentences might add around +0.5%. That gain only holds if my hand-written sentences match the real data distribution. My plan is to write about 600 to 1000 labeled sentences covering all 8 classes, several lines of business, and different registers.

Now I'm working out the feature reads for my sentences: pulling the layer-11 CLS and mean-pooled hidden states from a base pass over texts up to 1024 tokens. I'm also checking that the mean includes CLS and SEP, since the earlier edit verification depends on that matching.

I'm thinking about how many labeled sentences to write, since the gain depends on matching the real distribution. Writing around 800 by hand is a lot, so I'm leaning toward 60 to 80 per class. Before writing anything, I want to look at some practice examples in each class to match their style, especially for the confusable pairs.



The learning curve still rises from 897 to 1197 training sentences, so I'll write several hundred labeled insurance sentences and test whether they help the kernel-center CV before rebuilding the FFN. First I'm checking practice examples per class to match the labeling conventions.



## tool_use: run_python

```json
{
 "code": "pr=practice()\nrng=np.random.default_rng(1)\nfor c in LABELS:\n    items=[p for p in pr if p['label']==c]\n    print('=====',c)\n    for p in rng.permutation(len(items))[:7]: print('  -', items[p]['text'][:170])\n"
}
```

