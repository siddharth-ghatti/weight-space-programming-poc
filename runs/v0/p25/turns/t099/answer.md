## reasoning summary

I'm sorting out which of the failures are real model errors, like the negation misses on "not covered" phrasing. The practice-exclusion wording may be the difference, and these hand-written examples are what teach the model here. I'm also starting to look at the premium figure in the health plan text.

I'm noticing the model reads dollar amounts like "$512 per month" as a deductible signal, which is a shortcut worth flagging. Since the test set follows the practice distribution, more hand-written examples in practice-style phrasing should help most. I'll pull a sample of practice sentences to see how they're phrased.



Several disagreements are clear model errors, such as coverage exclusions predicted as coverage_grant and a premium amount predicted as deductible_limits, so targeted extra data can help. I'm now looking at practice-set style per register so the new sentences better match the test distribution.



## tool_use: run_python

```json
{
 "code": "\npr=practice(); rng=np.random.default_rng(7)\nfor i in rng.choice(len(pr),48,replace=False): print(f\"[{pr[i]['label'][:12]}] {pr[i]['text']}\")\n"
}
```

