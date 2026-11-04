## reasoning summary

Pass 46 is finished; next I'll check the logits against the measured-feature pool and the prediction, the new carry, feature drift, and training accuracy and margins.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nimport numpy as np, json\nfrom lib10 import *\nfrom lib7 import labels_for\nids=json.load(open('pass_46/meta.json'))['ids']; allids=[r['id'] for r in practice()]; assert ids==allids[:1024]\ny=labels_for(allids, practice, LABELS)[:1024]\nlg=np.load('pass_46/logits.npy').astype(np.float64)\nMa=np.load('pass_46/mean_hidden_L7.npy').astype(np.float64); Ca=np.load('pass_46/cls_hidden_L7.npy').astype(np.float64)\nd=np.load('final_design.npz'); Mn,Cn=d['Mn'][:1024],d['Cn'][:1024]\npl=np.load('pred_logits_final.npy')[:1024]\nla=pool_logits(Ma,Ca,W)\nprint('actual acc', (lg.argmax(1)==y).mean(), ' | pool-from-actual-feats max diff', np.abs(la-lg).max(), ' | vs prior prediction max diff', np.abs(pl-lg).max(), 'argmax agree', (pl.argmax(1)==lg.argmax(1)).mean())\nf=np.load('feats_drop46.npz'); Mo,Co=f['M'][:1024],f['C'][:1024]\nrel=lambda a,b: np.linalg.norm(a-b)/np.linalg.norm(b)\nprint('KEEP drift M', rel(Ma[:,KEEP],Mo[:,KEEP]), 'C', rel(Ca[:,KEEP],Co[:,KEEP]))\nprint('RES pred err M', rel(Ma[:,RES],Mn[:,RES]), 'C', rel(Ca[:,RES],Cn[:,RES]))\nsrt=np.sort(lg,1); mg=srt[:,-1]-srt[:,-2]; print('min margin', mg.min().round(4), '5pct', np.percentile(mg,5).round(4))\nprint('logit range', lg.min().round(3), lg.max().round(3))"
}
```

