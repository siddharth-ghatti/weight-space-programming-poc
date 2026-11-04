## tool_use: run_python

```json
{
 "code": "import json\nP = practice()\nids = [p['id'] for p in P]\njson.dump(ids, open(WORK/'ids.json','w'))\nprint(len(ids), ids[:3], ids[1020:1026], (WORK/'notes.md').exists())\n"
}
```

