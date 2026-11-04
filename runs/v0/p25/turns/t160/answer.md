## reasoning summary

I'm checking that the my3 sentences line up in order, and confirming the count of 467 from my notes matches what's actually there after deduplication. I also need to make sure the labels correspond correctly.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0,'.')\nfrom kh import *\nD=load(); d5=np.load('my5_feats_clean.npz')\nZp,yp=D['Zp'],D['yp']\ntm=json.load(open('my_sents_all.json')); t3=json.load(open('my_sents3_all.json')); t4=json.load(open('my_sents4_all.json'))\nfor t,y in [(tm,D['ym']),(t3,D['y3']),(t4,D['y4'])]: print((np.array(t['labels'])==y).mean())\ntexts=tm['texts']+t3['texts']+t4['texts']+list(d5['texts'])\nZmine=np.concatenate([D['Zm'],D['Z3'],D['Z4'],d5['Z']]); ymine=np.concatenate([D['ym'],D['y3'],D['y4'],d5['y']])\nM=len(Zmine); rng=np.random.default_rng(0); fo=rng.permutation(M)%5\npred=np.zeros(M,int); marg=np.zeros(M)\nfor f in range(5):\n    tr=fo!=f\n    Zt=np.concatenate([Zp,Zmine[tr]]); yt=np.concatenate([yp,ymine[tr]]); wt=np.concatenate([np.ones(1496),np.full(tr.sum(),0.4)])\n    a,b=fit(gelu(Zt@Zt.T/768),yt,wt,0.005)\n    F=gelu(Zmine[~tr]@Zt.T/768)@a+b; pred[~tr]=F.argmax(1)\n    marg[~tr]=F.max(1)-F[np.arange((~tr).sum()),ymine[~tr]]\nprint('OOF acc on mine', (pred==ymine).mean())\nbad=np.where(pred!=ymine)[0]; bad=bad[np.argsort(-marg[bad])]\njson.dump({'idx':bad.tolist(),'pred':pred[bad].tolist(),'marg':marg[bad].tolist()},open('mine_oof_bad.json','w'))\nfor i in bad[:45]: print(i, round(marg[i],2), LABELS[ymine[i]][:10],'->',LABELS[pred[i]][:10],'|',texts[i][:120])\n"
}
```

