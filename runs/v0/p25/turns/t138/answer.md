## reasoning summary

The new points help a little: non-center training points score 0.9913, against 0.9924 for centers. So adding more sentences as non-centers, or swapping some centers for them, might give a small gain. Many of the remaining errors look ambiguous or like label noise, such as a "free-look refund" labeled as term_cancellation.

I'm going through the error patterns that new sentences could target. Several look like label confusion: notice-of-loss wording gets pulled toward conditions_duties instead of claims_procedure, and a few other phrasings are similarly mislabeled.

I'm weighing whether the pooler could carry more of the capacity: it currently uses only 8 rows, and adding 760 tanh units on the CLS hidden state would mostly be random features of the same signal. A smarter option might be to pick the Nyström centers more carefully, since extra training points add roughly two-thirds of the benefit that centers do.

I'm weighing whether the Nyström centers could be chosen more carefully, since the spare neurons could also be used to drop redundant points from my batch. Layer 11's attention output is another place extra capacity might come from, but I haven't worked that through yet.

I'm weighing whether the Nyström centers could be trimmed, since the layer 10 and layer 11 FFNs already use all their neurons for the base features. Changing layer 10 would shift the features I'm building on, so I'm leaning toward keeping the architecture fixed and putting the effort into more training data.

Next I'll write the my5 batch, roughly 500 sentences split across several runs, each run producing about 125 sentences as JSON. Before that, I want to check the format of my_sents4_all.json and look at a few examples so the new sentences stay varied and don't duplicate what's already there.



Non-center training points get about two-thirds of the center benefit, so another batch of my own sentences is still worth adding. The remaining CV errors cluster around notice of loss, subrogation, accelerated-benefit charges, coverage grants and free-look refunds, and I'm checking the format of my earlier batches first.



## tool_use: run_python

```json
{
 "code": "import json\na=json.load(open('my_sents4_all.json')); print(type(a), len(a), a[:3] if isinstance(a,list) else list(a.items())[:1])\nb=json.load(open('my_sents3_all.json')); print(len(b), b[:2])\nimport collections\nprint(collections.Counter([x['label'] if isinstance(x,dict) else x[1] for x in a]))\n"
}
```

