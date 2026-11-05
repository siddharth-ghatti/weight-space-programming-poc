## tool_use: run_python

```json
{
 "code": "import os\nfor L in range(6):\n    for f in [f'ffn_in_L{L}.npy', f'ffn_ed_L{L}_B1.npy', f'ffn_ed_L{L}_B2.npy', f'ffn_in_L{L}_cls.npy', f'ffn_ed_L{L}_cls.npy']:\n        if os.path.exists(f): print(f, np.load(f, mmap_mode='r').shape)\nprint(json.load(open('batches.json')).keys() if isinstance(json.load(open('batches.json')),dict) else len(json.load(open('batches.json'))))\n"
}
```

