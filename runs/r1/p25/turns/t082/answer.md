## tool_use: run_python

```json
{
 "code": "\nm1=load(1); lab1=np.array(m1['labels'])\nLb=np.load('pass_1/logits.npy')\ndef stats(k):\n    m=load(k); assert m['ids']==m1['ids']\n    L=np.load(f'pass_{k}/logits.npy')\n    pb=Lb.argmax(1); pe=L.argmax(1)\n    agree=(pb==pe).mean()\n    srt=np.sort(Lb,1); marg=srt[:,-1]-srt[:,-2]; low=marg<np.quantile(marg,0.2)\n    r2=1-((L-Lb)**2).sum()/((Lb-Lb.mean(0))**2).sum()\n    acc=(pe==lab1).mean()\n    from collections import Counter\n    # macro F1\n    f=[]\n    for c in range(4):\n        tp=((pe==c)&(lab1==c)).sum(); fp=((pe==c)&(lab1!=c)).sum(); fn=((pe!=c)&(lab1==c)).sum()\n        f.append(2*tp/(2*tp+fp+fn))\n    out=[]\n    for l in range(7):\n        try:\n            hb=np.load(f'pass_1/cls_hidden_L{l}.npy'); he=np.load(f'pass_{k}/cls_hidden_L{l}.npy')\n            out.append(round(1-((he-hb)**2).sum()/((hb-hb.mean(0))**2).sum(),4))\n        except Exception as e: out.append(None)\n    print(k, 'agree',agree,'low',(pb==pe)[low].mean(),'R2',round(r2,4),'acc',acc,'F1',np.round(np.mean(f),4),'clsR2',out)\nprint('base acc',(Lb.argmax(1)==lab1).mean())\nfor k in [12,13,17]: stats(k)\n"
}
```

