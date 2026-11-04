## tool_use: run_python

```json
{
 "code": "print(open('cvlib.py').read()); print('-----'); print(open('prunelib.py').read()[:3000]); print(json.load(open('ids.json'))[:3] if isinstance(json.load(open('ids.json')),list) else list(json.load(open('ids.json')).keys()))"
}
```

