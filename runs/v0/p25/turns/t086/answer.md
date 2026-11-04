## tool_use: run_python

```json
{
 "code": "\nimport json\nfrom scipy.special import erf\nm=json.load(open('my_sents_all.json')); print(type(m), list(m.keys())[:5] if isinstance(m,dict) else None)\nfor kk in m: print(kk, type(m[kk]), (len(m[kk]) if hasattr(m[kk],'__len__') else m[kk]), str(m[kk])[:200])\nprint(os.listdir('pass_10'))\n"
}
```

