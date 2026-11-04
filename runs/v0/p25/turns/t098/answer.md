## reasoning summary

I'm looking at how the error drops as the training set grows: from 478 to 1197 examples, error falls from about 5.7% to 2.0%. Extrapolating to the full 1496 examples used in the final model suggests roughly 1.6%, and adding in-distribution data should help a bit more, so I'm leaning toward training the final model on everything.

I'm weighing whether to add about 500 sentences of my own that match the practice style more closely, since my earlier generic phrasing was likely the reason the model and my labels disagreed. Writing around 400 of them is a sizable but workable job across several passes, and the training set would then grow to roughly 1,500 plus about 1,000 new examples.

I'm considering an augmentation approach that needs no new writing: swapping company names, amounts, and lines of business in the practice sentences to teach the model those details don't matter. That would mostly produce near-duplicates of existing centers, though, so the gain is uncertain. A more useful next step may be checking which of my own sentences the practice model gets wrong, since those hard cases would show whether my labels or the model are off.

I'm weighing whether a few hundred more handwritten sentences are worth the effort, since the gain from my existing ones was only about 0.2%, which may not hold up on the test set. Testing whether the model works with the mean features alone, without the CLS features, could simplify things before I commit to that work.



Out-of-fold error keeps falling as training data grows (5.7% to 2.0%), so the final model should land around 98.3–98.5%. Next I'll check what `cls11` contributes and how my sentences differ from the practice data.



## tool_use: run_python

```json
{
 "code": "\nimport json\nfrom scipy.special import erf\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\nd=np.load('base_feats.npz'); y=d['y']; n=len(y)\ndef ln(x): x=x-x.mean(1,keepdims=True); return x/np.sqrt((x**2).mean(1,keepdims=True)+1e-12)\ndef stdz(X): return (X-X.mean(0))/(X.std(0)+1e-6)\ndef cvk(Z, lam=0.01, beta=1.0, seeds=4):\n    K=gelu(beta*Z@Z.T/Z.shape[1]); accs=[]\n    for seed in range(seeds):\n        f=np.random.default_rng(seed).permutation(n)%5; P=np.zeros((n,8))\n        for j in range(5):\n            tr=f!=j; te=~tr; T=np.eye(8)[y[tr]]; b=T.mean(0)\n            A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()),T-b); P[te]=K[np.ix_(te,tr)]@A+b\n        accs.append((P.argmax(1)==y).mean())\n    return round(np.mean(accs),4)\nprint('mean11 only', cvk(stdz(ln(d['mean11']))))\nprint('cls11+2mean11', cvk(stdz(ln(d['cls11']+2*d['mean11']))))\nprint('cls11+4mean11', cvk(stdz(ln(d['cls11']+4*d['mean11']))))\nm=json.load(open('my_sents_all.json')); mp=np.load('my_pred_by_practice.npy'); ml=np.array(m['labels'])\nprint(mp.shape, (mp==ml).mean() if mp.ndim==1 else None)\nimport collections\nprint(collections.Counter(ml.tolist()))\ndis=np.where(mp!=ml)[0]\nfor i in dis[:45]: print(f\"[{LABELS[ml[i]][:10]} vs {LABELS[mp[i]][:10]}] {m['texts'][i]}\")\n"
}
```

