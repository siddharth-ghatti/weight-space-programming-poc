## reasoning summary

I'm starting each run from a clean slate, and I need to write a helper module for that.



## tool_use: run_python

```json
{
 "code": "\nsrc = r'''\nimport json, numpy as np\ndef load(k): return json.load(open(f'pass_{k}/meta.json'))\nm1=load(1); lab1=np.array(m1['labels'])\nLb=np.load('pass_1/logits.npy')\ndef f1(pe,lab):\n    f=[]\n    for c in range(4):\n        tp=((pe==c)&(lab==c)).sum(); fp=((pe==c)&(lab!=c)).sum(); fn=((pe!=c)&(lab==c)).sum()\n        f.append(2*tp/(2*tp+fp+fn))\n    return np.mean(f)\ndef stats(k, L=None, verbose=True):\n    if L is None:\n        m=load(k); assert m['ids']==m1['ids']\n        L=np.load(f'pass_{k}/logits.npy')\n    pb=Lb.argmax(1); pe=L.argmax(1)\n    srt=np.sort(Lb,1); marg=srt[:,-1]-srt[:,-2]; low=marg<np.quantile(marg,0.2)\n    r2=1-((L-Lb)**2).sum()/((Lb-Lb.mean(0))**2).sum()\n    out=[]\n    if k is not None:\n      for l in range(7):\n        try:\n            hb=np.load(f'pass_1/cls_hidden_L{l}.npy'); he=np.load(f'pass_{k}/cls_hidden_L{l}.npy')\n            out.append(round(1-((he-hb)**2).sum()/((hb-hb.mean(0))**2).sum(),4))\n        except Exception as e: out.append(None)\n    d=dict(agree=(pb==pe).mean(),low=(pb==pe)[low].mean(),R2=round(r2,4),acc=(pe==lab1).mean(),F1=round(f1(pe,lab1),4),clsR2=out)\n    if verbose: print(k,d)\n    return d\n'''\nopen('evalh.py','w').write(src)\nexec(src)\nprint('base acc',(Lb.argmax(1)==lab1).mean(), f1(Lb.argmax(1),lab1))\nfor k in [12,13,17]: stats(k)\n"
}
```

