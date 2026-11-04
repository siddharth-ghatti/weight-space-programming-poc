## tool_use: run_python

```json
{
 "code": "import json, sys; sys.path.insert(0,'.'); import readout\npr=json.load(open('pass_10/predictions.json'))\nF=np.load('feats.npz'); y=F['y']; R=dict(np.load('readout_L9.npz'))\nidx={p['id']:i for i,p in enumerate(practice())}\nii=np.array([idx[p['id']] for p in pr]); Lm=np.arctanh(np.clip(np.array([p['logits_edited'] for p in pr]),-0.9999,0.9999))\nLo=readout.logits_offline(R,F['C9'][ii],F['M9'][ii])\na=Lm-Lm.mean(1,keepdims=True); b=Lo-Lo.mean(1,keepdims=True)\nratio=(a*b).sum(1)/(b*b).sum(1); resid=a-ratio[:,None]*b\nprint('ratio range',ratio.min().round(3),ratio.max().round(3),' max rel resid',(np.abs(resid).max(1)/np.abs(a).max(1)).max().round(4))\nprint('argmax agree', (Lm.argmax(1)==Lo.argmax(1)).mean(), ' offline acc on these', (Lo.argmax(1)==y[ii]).mean())\nprint('max |logit|', np.abs(np.array([p['logits_edited'] for p in pr])).max())"
}
```

