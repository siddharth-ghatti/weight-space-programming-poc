## reasoning summary

I need to keep the output small, so I'll compute the correlation between the absolute value of dl and length.



## message

That printed far too much (I'll avoid dumping files like this). Now the actual check, logit error by length bucket:

## tool_use: run_python

```json
{
 "code": "exec(open('common.py').read())\nL=json.load(open('lens.json')); ids=json.load(open('all_ids.json'))\nln_={i:l for i,l in zip(L['ids'],L['lens'])}; lens=np.array([ln_[i] for i in ids])\nlg=np.load('lg_pass26.npy'); d=np.abs(lg-BL).mean(1)\nfor lo,hi in [(0,60),(60,90),(90,130),(130,200),(200,600)]:\n    s=(lens>=lo)&(lens<hi); print(lo,hi,s.sum(),'mean|dl| %.3f'%d[s].mean(), 'flips',(lg.argmax(1)!=bp)[s].sum())\nprint('corr', np.corrcoef(np.log(lens), d)[0,1])\n"
}
```

