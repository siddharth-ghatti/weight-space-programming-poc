## tool_use: run_python

```json
{
 "code": "\ns=open('evalh.py').read().replace(\"lab1=np.array(m1['labels'])\",\"lab1=np.load('y.npy')\")\nopen('evalh.py','w').write(s)\nfrom evalh import *\nprint('base acc',(Lb.argmax(1)==lab1).mean(), f1(Lb.argmax(1),lab1))\nfor k in [12,17]: stats(k)\n"
}
```

