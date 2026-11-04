## tool_use: run_python

```json
{
 "code": "import json\nids=[x['id'] for x in practice()]\njson.dump(ids, open(WORK/'ids.json','w'))\nprint(ids[:3], ids[1020:1026], (WORK/'notes.md').exists())\n"
}
```

