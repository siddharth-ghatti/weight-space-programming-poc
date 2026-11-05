## tool_use: run_python

```json
{
 "code": "\nimport json\nm=json.load(open('pass_1/meta.json')); print(m['labels'][:5]); print(np.load('y.npy')[:5], np.load('y.npy').shape)\nP=practice(); d={p['id']:p['label'] for p in P}; print([d[i] for i in m['ids'][:5]])\n"
}
```

