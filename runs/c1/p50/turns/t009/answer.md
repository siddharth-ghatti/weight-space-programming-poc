## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\nd=load_base(); y=labels()\nfor l in range(13):\n    rm=cv_acc(d[f'mean_hidden_L{l}'][:,KEEP],y,lams=(1,10,100,1000))\n    rc=cv_acc(d[f'cls_hidden_L{l}'][:,KEEP],y,lams=(1,10,100))\n    print(l,'mean',{k:round(v,4) for k,v in rm.items()},'cls',{k:round(v,4) for k,v in rc.items()})"
}
```

