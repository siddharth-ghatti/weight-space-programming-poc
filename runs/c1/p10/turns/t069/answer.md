## reasoning summary

The in-sample margin minimum came out at 0.76, which is large and may point to overfitting, though it is still only in-sample. Next I want to check it with a read pass: running the edited model to get logits for the first 1024 ids, possibly the pooled version too, and comparing them against the expected logits. Before using any pass, I'll update the notes.



The head is written and matches KRR on every sentence in float32, with a smallest top-1 margin of 0.76. Next I'll update the notes and check the edited model's logits and last two hidden layers for 1024 ids against my numpy expectation.



## tool_use: run_python

```json
{
 "code": "s=open('notes.md').read()\ns+='''\n## Head built (run 64): layer-11 FFN = GELU-kernel KRR. kappa(u)=GELU(u), u=z(x1).z(c_i)/760 (z-score over KEEP dims, mu/sd from all 1496),\n  lam=0.01, units 0..1495 = practice centers (order = base_feats/ids.json), W2 -> RES rows = alpha^T, LN2 gamma=1 all, pooler rows0-7 = (e_k-1/8) on RES, bias0.\n  CV: 20-fold 98.53, 5-fold x4 98.13 (RBF 98.7/98.2). X1_full.npy = exact pooling-layer FFN input; expected_logits.npy saved.\n- Next: pass_6 read edited logits ids[:1024] -> compare to expected_logits (should be ~1e-4).\n'''\nopen('notes.md','w').write(s)\nids=json.load(open('ids.json')) if 'json' in dir() else None\nimport json; ids=json.load(open('ids.json')); print(ids[:2], len(ids), structure()['layers'], structure()['m'])\n"
}
```

## tool_use: forward_pass

```json
{
 "kind": "read",
 "model": "edited",
 "ids": [
  "p0000",
  "p0001",
  "p0002",
  "p0003",
  "p0004",
  "p0005",
  "p0006",
  "p0007",
  "p0008",
  "p0009"
 ],
 "outputs": [
  "logits"
 ],
 "layers": [
  10
 ]
}
```

