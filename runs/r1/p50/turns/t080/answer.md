## reasoning summary

I'm checking how the shift model's margins put many test items near zero, which would make them high-risk under any perturbation, and noting that it still predicts a mean loss of 1.4 even at m2048. The tau model looks more plausible to me, with lower risk. I'm also starting to think about a bias where more test items sit near the boundary on the correct side than the wrong side.

I'm weighing whether a smaller model can hit a noise std near 0.2, since the 6-layer m2048 version gets about 0.23 at 72.9M parameters. I'm wondering whether the new data might push that closer to 0.2 and whether m1792 could land near 0.28. I'm also considering whether measuring the noise contribution of each layer, by pruning one at a time at m=1024, would show which layers are worth improving.

I'm wondering whether CLS rows, which are only a small share of the probe data, carry disproportionate error since they're atypical with a large constant dimension. I'm considering checking per-token-type error in holdout, and if CLS rows are much worse, giving them dedicated weighting in the fit.

I'm planning to check whether the error is concentrated in the CLS and </s> rows compared with the other tokens, measured relative to each row's own scale. If CLS rows turn out to be much worse, I'd consider collecting more CLS samples per layer and giving them dedicated weight in the fit.



Both test models put the failure risk in the double digits or low single digits, and a margin noise std near 0.2 is clearly safer than 0.35, so I need less noise per parameter. I'm checking the held-out error by token type next, since special sink tokens like `<s>` and `</s>` may carry an outsized share of the error.



## tool_use: run_python

```json
{
 "code": "exec(open('seq.py').read())\n# token types: need token ids for old probe data and B1/B2. Build from tokenizer.\npid = json.load(open(WORK/'probe_ids.json')); bt = json.load(open(WORK/'batches.json'))\nP = {p['id']:p['text'] for p in practice()}\ndef tok_types(idl):\n    out=[]\n    for t in tokenize([P[i] for i in idl]):\n        ids_ = t['ids'][:512]; tt = np.zeros(len(ids_),int); tt[0]=1; tt[-1]=2\n        out.append(tt)\n    return out\n# old data order: check concatenation length\ntt_old = np.concatenate(tok_types(pid)); print(len(tt_old))\ntt_B1 = np.concatenate(tok_types(bt['B1'])); tt_B2 = np.concatenate(tok_types(bt['B2'])); print(len(tt_B1), len(tt_B2))\nTT = np.r_[tt_old, tt_B1, tt_B2]; np.save(WORK/'tok_types_all.npy', TT)\nfor L in [1,3,4]:\n    A, S = all_data(L); te = (S<100)&(S%2==1); tr=~te\n    keep,B,mk,mt = fit_layer(L, 1536, 0.05, A=A[tr], save=False)\n    pre = f\"roberta.encoder.layer.{L}.\"\n    W1 = base(pre+\"intermediate.dense.weight\").astype(np.float64); b1 = base(pre+\"intermediate.dense.bias\")\n    W2 = base(pre+\"output.dense.weight\").astype(np.float64); b2 = base(pre+\"output.dense.bias\")\n    g2 = base(pre+\"output.LayerNorm.weight\")\n    A = A.astype(np.float64); H = gelu(A@W1.T+b1); Z = A + H@W2.T + b2; sig = Z.std(1)\n    drop = np.setdiff1d(np.arange(3072), keep)\n    err = H[:,drop]@W2[:,drop].T - ((H[:,keep]-mk)@B + mt)\n    post = np.sqrt((((err-err.mean(1,keepdims=True))*g2/sig[:,None])**2).sum(1))\n    for k,nm in enumerate(['tok','<s>','</s>']):\n        msk = (TT==k)\n        print(L, nm, 'n', msk.sum(), 'post-LN err train', post[msk&tr].mean().round(3), 'test', post[msk&te].mean().round(3), 'sigma', sig[msk].mean().round(2))"
}
```

