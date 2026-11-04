## tool_use: run_python

```json
{
 "code": "import json\nlg=np.load('pass_8/logits.npy'); m=json.load(open('pass_8/meta.json')); ids=json.load(open('ids.json'))\nix=np.array([ids.index(i) for i in m['ids']]); ex=np.load('expected_logits.npy')[ix]\ny=np.load('base_feats.npz')['y'][ix]\nprint('max abs err', np.abs(lg-ex).max(), 'argmax agree', (lg.argmax(1)==ex.argmax(1)).mean(), 'acc vs labels (in-sample)', (lg.argmax(1)==y).mean())\ns=open('notes.md').read()\ns+=f'''\n## pass_8: edited logits p0472..p1495 vs expected: max err {np.abs(lg-ex).max():.2e}, argmax agree {(lg.argmax(1)==ex.argmax(1)).mean()}. READY TO SUBMIT (2 passes left).\n'''\nopen('notes.md','w').write(s)\n"
}
```

