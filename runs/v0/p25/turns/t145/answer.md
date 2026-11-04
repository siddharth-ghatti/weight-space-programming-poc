## reasoning summary

I'm weighing whether the insurer's right to examine an injured claimant fits under claims procedure. There's a possible overlap with what I already put in c4 about independent medical exams, so I'm checking whether these two items conflict. I'm also looking at how standard health-policy "physical examination" clauses are handled in practice.

I'm also wondering whether "examinations under oath" from the homeowners "Duties After Loss" section gets classified as claims procedure, so I want to check how that is labeled in practice, along with the other examination and records language.



## message

A few of my conditions_duties sentences overlap claims topics (medical examinations, examination under oath, protecting property after a loss, keeping records). I'll check how practice labels these.

## tool_use: run_python

```json
{
 "code": "pr=practice()\nfor kw in ['examin','under oath','further damage','protect the property','records','reasonable steps','legal action','suit against','mediat','governed by','contest']:\n    hits=[(p['label'], p['text'][:130]) for p in pr if kw in p['text'].lower()]\n    import collections\n    print('==',kw, collections.Counter(h[0] for h in hits))\n    for h in hits[:3]: print('   ',h)\n"
}
```

