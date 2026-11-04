## message

Now I'll align the batch-4 rows, save the features, and run CV with all three hand-written batches.

## tool_use: run_python

```json
{
 "code": "\nimport json\nfrom scipy.special import erf\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\ndef ln(x): x=x-x.mean(1,keepdims=True); return x/np.sqrt((x**2).mean(1,keepdims=True)+1e-12)\nk=np.load('kernel_v3.npz'); mu,sd=k['mu'],k['sd']\nd4=json.load(open('my_sents4_all.json')); mt=json.load(open('pass_13/meta.json'))['texts']\nprint('mismatch', sum(a!=b for a,b in zip(mt,d4['texts'])))\nc=np.load('pass_13/cls_hidden_L11.npy'); m=np.load('pass_13/mean_hidden_L11.npy')\nZ4=(ln(c+2*m)-mu)/sd; y4=np.array(d4['labels']); np.savez('my4_feats.npz',Z=Z4,y=y4)\nwith open('notes.md','a') as f: f.write(\"- my_sents4_all.json: 480 more (informal/broker/letter style). Feats: my3_feats.npz (pass_12, 467 after dedup), my4_feats.npz (pass_13). Z already standardized w/ kernel_v3 mu/sd.\\n- CV with my3 added (w=.3, lam .02-.05): 0.9875-0.988 vs practice-only 0.980. Big win.\\n\")\nf3=np.load('my3_feats.npz')\nC=k['C'].astype(np.float64); yc=k['yc']; n=1496\nCall=np.vstack([C,f3['Z'],Z4]).astype(np.float64); yall=np.concatenate([yc,f3['y'],y4])\nsrc=np.concatenate([np.zeros(n),np.ones(len(yc)-n),2*np.ones(len(f3['y'])),3*np.ones(len(y4))]).astype(int)\nK=gelu(Call@Call.T/768)\ntr=np.arange(n); Tm=np.eye(8)[yc[:n]]; b=Tm.mean(0)\nA=np.linalg.solve(K[np.ix_(tr,tr)]+0.01*np.eye(n),Tm-b); p4=(K[np.ix_(np.where(src==3)[0],tr)]@A+b).argmax(1)\nprint('practice-model agreement on batch4', (p4==y4).mean(), [round((p4[y4==c_]==c_).mean(),2) for c_ in range(8)])\ndef cv(ws,lam,seeds=6):\n    accs=[]\n    for seed in range(seeds):\n        f=np.random.default_rng(seed).permutation(n)%5; P=np.zeros((n,8))\n        for j in range(5):\n            trp=np.where(f!=j)[0]; te=np.where(f==j)[0]; tr=np.concatenate([trp, np.where(src>0)[0]])\n            w=np.array([1.0,*ws])[src[tr]]; tr=tr[w>0]; w=w[w>0]\n            Tm=np.eye(8)[yall[tr]]; b=(w[:,None]*Tm).sum(0)/w.sum()\n            A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.diag(1/w),Tm-b); P[te]=K[np.ix_(te,tr)]@A+b\n        accs.append((P.argmax(1)==yall[:n]).mean())\n    return round(np.mean(accs),4)\nfor ws in [(0.3,0.3,0),(0.3,0.3,0.3),(0.3,0.5,0.5),(0.2,0.4,0.4),(0.3,0.3,0.6)]:\n    print(ws,[cv(ws,lam) for lam in [0.02,0.03,0.05]])\n"
}
```

