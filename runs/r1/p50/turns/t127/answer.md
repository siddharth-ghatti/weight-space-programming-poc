## reasoning summary

I'm noting that passes 27 and 28 are only partial noise simulations, covering just layers 0–1 and layers 2–4 respectively. Next I'll look at which files are present in pass_12.



## tool_use: run_python

```json
{
 "code": "import os\nfor p in ['pass_10','pass_11','pass_12','pass_24','pass_25']:\n    print(p, os.listdir(p))\n"
}
```

