## reasoning summary

I'm checking how the edited model at m=1024 and the m=1536 predictions agree with the base model, starting by turning the pass-24 layer-6 features into logits through the classifier head.

I'm checking whether the pass-24 and pass-25 logits agree with the base model's, and there's an existing lg_m1536_v2.npy file I can use for the m=1536 side.



Passes 13–23 were sequential probes, pass 24 read the m=1024 model's final `<s>` vectors, and pass 25 tested m=1536. Next I'll measure how often each model's prediction matches base.



## tool_use: run_python

```json
{
 "code": "exec(open('common.py').read()) if os.path.exists('common.py') else None\nimport json\nbl=np.load('base_logits.npy'); y=np.load('y.npy'); ids=json.load(open('all_ids.json'))\nbp=bl.argmax(1)\ndef head(h):\n    p=np.tanh(h@W('classifier.dense.weight').T+W('classifier.dense.bias'))\n    return p@W('classifier.out_proj.weight').T+W('classifier.out_proj.bias')\nm24=json.load(open('pass_24/meta.json'))\nassert m24['ids']==ids\nlg24=head(np.load('pass_24/cls_hidden_L6.npy'))\nfrom collections import Counter\ndef f1(y,p):\n    fs=[]\n    for c in range(4):\n        tp=((p==c)&(y==c)).sum(); fp=((p==c)&(y!=c)).sum(); fn=((p!=c)&(y==c)).sum()\n        fs.append(2*tp/(2*tp+fp+fn))\n    return np.mean(fs)\nprint('m1024 seq: agr',(lg24.argmax(1)==bp).mean(),'acc',(lg24.argmax(1)==y).mean(), 'f1',f1(y,lg24.argmax(1)))\npr=json.load(open('pass_25/predictions.json'))\nprint(type(pr), (pr[0] if isinstance(pr,list) else list(pr.keys())))\n"
}
```

