## tool_use: run_python

```json
{
 "code": "import json\nids=[p['id'] for p in practice()]; print(len(ids), ids[0], ids[1023], ids[1024], ids[-1])\nprint(all(ids[i]==f'p{i:04d}' for i in range(len(ids))))"
}
```

