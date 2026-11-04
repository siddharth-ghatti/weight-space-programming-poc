## reasoning summary

The validation score is already at 0.9947, so the remaining misses are mostly ambiguous cases. "Prompt notice of loss" is one: it could be a claims procedure or a policy condition, since duties after loss usually count as conditions. Adding more sentences about notice of loss as a claims procedure might help, but the test labels may be noisy enough that it's not worth chasing.

I'm weighing one more targeted round of about 250 sentences on the confused pairs, like notice-of-loss versus claims, and refund versus cancellation. They'd be non-center points, so the expected CV gain is small, maybe around 0.1%. Capacity doesn't seem to be the limit either, since random and pivoted centers scored the same, so the data is what's binding.

I'm going to check how the practice data labels a few boundary phrases, like "notice", "refund", and "cooperate", so my sentences match its conventions. Prompt notice of loss already looks like claims_procedure, so I'll keep that consistent.



Cross-validation puts the v5 setup at 0.9947 accuracy, with the remaining errors clustered on boundary pairs such as claims vs. conditions and premium vs. cancellation. I'm checking how the practice data labels those boundary phrases so a small targeted batch follows the same conventions.



## tool_use: run_python

```json
{
 "code": "import re\nP=practice()\ndef grep(pat, k=12):\n    hits=[p for p in P if re.search(pat,p['text'],re.I)]\n    from collections import Counter\n    print(pat, Counter(p['label'] for p in hits))\n    for p in hits[:k]: print('  ',p['label'][:12],'|',p['text'][:150])\ngrep(r'\\bnotice\\b',10)\ngrep(r'refund',10)\ngrep(r'cooperat',4)\ngrep(r'reimburse',6)\ngrep(r'\\bincludes?\\b.*\\bmeans?\\b|\\bincludes\\b',6)\n"
}
```

