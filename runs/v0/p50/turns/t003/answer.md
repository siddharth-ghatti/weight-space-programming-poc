## tool_use: run_python

```json
{
 "code": "from collections import Counter\nP = practice()\nprint(Counter(p['register'] for p in P)); print(Counter(p['line'] for p in P)); print(Counter(p['variant'] for p in P))\nopen('notes.md','w').write(\"\"\"# Notes\n## Strategy\n- Read hidden states (cls/mean, all layers) for all 1496 practice sentences (2 read passes).\n- Closed-form probe (ridge/LDA) with CV to pick the best feature.\n- Realize the feature at final CLS via weight edits (e.g. uniform CLS attention = mean pooling at a chosen layer,\n  later layers made pass-through), then put the probe into pooler+classifier.\n## Log\n- max token len 63, mean 31.\n\"\"\")\n"
}
```

