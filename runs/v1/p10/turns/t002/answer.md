## reasoning summary

I'm starting fresh and planning to write up notes with a strategy, then checking the data's statistics like token lengths and variants.



Fresh start: I'll read activations from the base model once, pick the best layer's features offline with closed-form classifiers, then reroute [CLS] to mean-pool and fit the head on the edited model.



## tool_use: run_python

```json
{
 "code": "import collections, json\nP = practice()\ntoks = tokenize([p['text'] for p in P])\nL = [len(t['ids']) for t in toks]\nprint('len stats', np.min(L), np.mean(L), np.max(L), np.percentile(L,[50,90,99]))\nprint(collections.Counter(p['line'] for p in P))\nprint(collections.Counter(p['register'] for p in P))\nprint(collections.Counter(p['variant'] for p in P))\nprint(toks[0]['tokens'][:10])\nnotes = \"\"\"# Notes\n## Strategy (run 1)\n- Read base activations (mean_hidden, cls_hidden all layers) for all 1496 practice ids (2 read passes).\n- Offline closed-form (ridge/LDA) CV per layer to pick features.\n- Edit last layer(s) so CLS attends uniformly (mean pool) -> read edited features -> fit pooler+classifier closed-form.\n\"\"\"\nopen('notes.md','w').write(notes)\n"
}
```

