## reasoning summary

The verified v3 result checks out at 1e-4 error, so that part is settled. Next I'm weighing two ways to push further: a second batch of about 400–500 sentences aimed at the confusable boundaries, where the first batch only added about 0.25%, or richer features for the kernel, perhaps drawing on the 12 heads of layer 11 attention alongside the current LN(cls11 + 2·mean11) feature.

I'm weighing whether to push the kernel further by richer features, such as using the layer 11 heads, but a linear-like term is already approximated by the gelu features, so that path looks less promising. A second batch aimed at practice-style text still seems the most likely route to better generalization, with the 0.3 weighting guarding against label-convention mismatch.

I'm now looking at where the v3 model still makes mistakes, mainly term versus premium and claims versus conditions, so I can target a second batch of about 480 practice-style sentences at those boundaries. I also want to make sure my labels follow the dataset's conventions, since that determines whether the extra data helps.

Next I'm weighing whether to write the cheaper variant, swapping company names and numbers in practice sentences, but those near-duplicates would add little beyond what the kernel centers already capture. I'll go with a second batch in the longer letter, FAQ, and broker-summary register, and I want to check how the token lengths of the practice sentences compare with mine so I can calibrate.



v3 checks out, with edited logits within 1e-4 of expected and a minimum practice margin of 6.1. Next I'm comparing length and register between practice and my sentences, and checking which CV confusions remain, so I can target a second writing batch.



## tool_use: run_python

```json
{
 "code": "import json, re\npr=practice(); A=json.load(open('my_sents_all.json'))\nlp=np.array([len(t['ids']) for t in tokenize([p['text'] for p in pr])]); lm=np.array([len(t['ids']) for t in tokenize(A['texts'])])\nprint('practice len pct', np.percentile(lp,[10,25,50,75,90]), 'mine', np.percentile(lm,[10,25,50,75,90]))\nq=np.mean(['?' in p['text'] for p in pr]); qm=np.mean(['?' in t for t in A['texts']]); print('frac with ?', q, qm)\ncaps=lambda t: len(re.findall(r'\\b[A-Z][a-z]+ (?:Mutual|Insurance|Life|Health|Casualty|Assurance)\\b', t))>0\nprint('frac w/ company name', np.mean([caps(p['text']) for p in pr]), np.mean([caps(t) for t in A['texts']]))\nrng=np.random.default_rng(5)\nfor i in rng.permutation(len(pr))[:25]: print('-', pr[i]['label'][:10], '|', pr[i]['text'])\n"
}
```

