## reasoning summary

I'm computing Z, checking that the texts line up, and saving my6_feats.npz. Next I'll see how often the v5 kernel's predictions on my6 disagree with the labels as a label-noise check, then run cross-validation with my6 included.



Read succeeded; next I'll compute Z for the new sentences, check the texts against my file, and save my6_feats.npz, then see how the v5 kernel handles them.



## tool_use: run_python

```json
{
 "code": "import sys, os; sys.path.insert(0, str(WORK)); os.chdir(WORK)\nimport numpy as np, json\nfrom kh import *\nout=json.load(open('my6_all.json'))\nmeta=json.load(open('pass_17/meta.json'))\nmt=meta.get('texts', None)\nprint(meta.keys() if isinstance(meta,dict) else type(meta))\nif mt: print('texts match', mt==[s['text'] for s in out])\nk=np.load('kernel_v5.npz')\nc=np.load('pass_17/cls_hidden_L11.npy'); m=np.load('pass_17/mean_hidden_L11.npy')\nx=c+2*m; ln=(x-x.mean(1,keepdims=True))/np.sqrt(x.var(1,keepdims=True)+1e-12)\nZ6=(ln-k['mu'])/k['sd']\nlab={l:i for i,l in enumerate(LABELS)}\ny6=np.array([lab[s['label']] for s in out])\nnp.savez('my6_feats.npz',Z=Z6,y=y6,texts=np.array([s['text'] for s in out]))\n# v5 kernel prediction\nF=gelu(Z6@k['C'].T/768)@k['A']+k['b0']\npr=F.argmax(1); print('v5 acc on my6', (pr==y6).mean())\nfrom collections import Counter\nprint(Counter((LABELS[a],LABELS[b]) for a,b in zip(y6,pr) if a!=b).most_common())\nfor i in np.where(pr!=y6)[0]: print(LABELS[y6[i]][:10],'->',LABELS[pr[i]][:10],'|',out[i]['text'])\n"
}
```

