## tool_use: run_python

```json
{
 "code": "import json\nids=[p['id'] for p in practice()]\njson.dump(ids, open('all_ids.json','w'))\nprint(len(ids), ids[:3], ids[1020:1026])"
}
```

## tool_use: forward_pass

```json
{
 "kind": "read",
 "model": "base",
 "ids": [
  "p0000"
 ],
 "outputs": [
  "token_hidden",
  "mean_hidden",
  "cls_hidden"
 ],
 "layers": [
  12
 ]
}
```

