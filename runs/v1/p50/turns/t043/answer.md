## reasoning summary

Pass_11 covered 1024 sentences, so 472 practice ids remain unseen; I'm running the edited model through the harness's predict path on those to check the pipeline agrees.



## tool_use: run_python

```json
{
 "code": "\nimport json; r=json.load(open('rest_ids.json')); print(json.dumps(r))\n"
}
```

