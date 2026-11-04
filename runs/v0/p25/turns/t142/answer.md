## reasoning summary

I'm checking the Waiver of Premium Rider wording in the practice's deductible limits, and I'm dropping two of my own waiver sentences because they read as ambiguous or coverage-like. Next I'll check how "notice of claim" is handled in the claims process.

I'm now comparing how the "notice" examples appear under claims_procedure and conditions_duties, looking at the three cases to see how each one is worded.



Practice labels are set: most conditions go to conditions_duties, reinstatement to term_cancellation_renewal, and the free-look and waiver sentences are dropped. Next I'm checking the three "notice" sentences under conditions_duties.



## tool_use: run_python

```json
{
 "code": "pr=practice()\nfor p in pr:\n    t=p['text'].lower()\n    if 'notice' in t and p['label'] in ('conditions_duties','premium_payment'): print(p['label'],'|',p['text'])\nprint()\nfor p in pr:\n    t=p['text'].lower()\n    if ('prompt' in t or 'at once' in t or 'immediately' in t or 'as soon as' in t): print(p['label'],'|',p['text'][:150])\n"
}
```

