## reasoning summary

I'll write up my notes first, then run forward passes to pull cls_hidden and mean_hidden for every layer, and maybe the pooled output too. After that I'll check the token lengths.



## message

Before spending forward passes, I'll start `notes.md` and check token lengths.

## tool_use: run_python

```json
{
 "code": "P = practice()\ntoks = tokenize([p['text'] for p in P])\nL = [len(t['ids']) for t in toks]\nprint(max(L), np.mean(L), toks[0]['tokens'][:5], toks[0]['tokens'][-3:])\nprint(Counter(p['register'] for p in P)); print(Counter(p['line'] for p in P)); print(Counter(p['variant'] for p in P))\nopen('notes.md','w').write(\"\"\"# Notes\n## Strategy\n- Read hidden states (cls/mean, all layers) for all 1496 practice sentences (2 read passes).\n- Closed-form probe (ridge/LDA) with CV to pick the best feature.\n- Realize the feature at final CLS via weight edits (e.g. uniform CLS attention = mean pooling at a chosen layer,\n  later layers made pass-through), then put the probe into pooler+classifier.\n## Log\n\"\"\")\n"
}
```

