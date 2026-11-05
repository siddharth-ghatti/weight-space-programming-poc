## tool_use: run_python

```json
{
 "code": "exec(open('common.py').read())\nids=json.load(open('all_ids.json')); idx={i:k for k,i in enumerate(ids)}\nlg=np.zeros((1000,4))\nfor r in json.load(open('pass_28/predictions.json')): lg[idx[r['id']]]=r['logits_edited']\nnp.save('lg_D2_L234.npy',lg); report('D2 L2-4 pruned', lg); print(' mean|dl| %.4f rms %.4f'%(np.abs(lg-BL).mean(), np.sqrt(((lg-BL)**2).mean())))\nd1=np.load('lg_D1_L01.npy')-BL; d2=lg-BL; dv=np.load('lg_pass26.npy')-BL\nprint('rms2: D1 %.4f D2 %.4f sum %.4f full %.4f'%((d1**2).mean(),(d2**2).mean(),(d1**2).mean()+(d2**2).mean(),(dv**2).mean()))\nprint('corr d1,d2', np.corrcoef(d1.ravel(), d2.ravel())[0,1], 'corr (d1+d2), full', np.corrcoef((d1+d2).ravel(), dv.ravel())[0,1])\n"
}
```

