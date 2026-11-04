## reasoning summary

I'm checking whether the perturbation from the CLS query is small enough to matter for attention. The average shift on used tokens is about 2%, so I'm wondering if its effect on the others is negligible. Then I'm trying to work out which vocabulary token has the 0.93 value, guessing it's [SEP], though I'm not sure yet since it's used throughout.

Next I'm checking whether SEP specifically has a large enough r·x to matter, since it shows up in every sentence and the period is common too. I'm also weighing whether CLS's layer-0 attention is worth keeping, and planning to read CLS at layer 1 to see if it still carries the mean-0 signal before going further.

I'm planning to run layer 0 first and read the CLS and mean hidden states across all layers, then compute the CV for the layer-11 outputs and check how the CLS signal changes per layer. For later layers I'd need token-level hidden states, and storing 64 texts by about 32 tokens across 13 layers is roughly 160MB, which seems manageable. I also want to check the r·x values for SEP at positions 10 to 63 and for the period.



The vector r gives r·e_CLS = 1 and only about 2% shift to other tokens' layer-0 queries, so I'm now boosting [SEP] and '.' and checking them before saving r.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nWemb=base('bert.embeddings.word_embeddings.weight').astype(np.float64)\nPemb=base('bert.embeddings.position_embeddings.weight').astype(np.float64)\nTemb=base('bert.embeddings.token_type_embeddings.weight').astype(np.float64)\ngam=base('bert.embeddings.LayerNorm.weight').astype(np.float64); bet=base('bert.embeddings.LayerNorm.bias').astype(np.float64)\nV=vocab(); cls_id=V['[CLS]']; sep_id=V['[SEP]']; dot=V['.']\nD=np.load('base_feats.npz'); eC=D['cls0'][0].astype(np.float64)\ntoks=tokenize([p['text'] for p in practice()])\nfreq=np.zeros(len(Wemb))\nfor t in toks:\n    for i in t['ids']: freq[i]+=1\nK=160\ndef rx(r,ids,pos):\n    v=Wemb[ids]+Pemb[pos]+Temb[0]\n    m=v.mean(1,keepdims=True); s=np.sqrt(((v-m)**2).mean(1,keepdims=True)+1e-12)\n    return (gam*(v-m)/s+bet)@r\ndef solve(sep_w, K=160, extra=[]):\n    C=np.vstack([np.ones(768), bet/gam, Temb[0], Pemb[1:K+1]]+extra)\n    U,S,Vt=np.linalg.svd(C); rank=(S>1e-8*S[0]).sum(); N=Vt[rank:].T\n    wts=freq/freq.sum()+0.2/len(Wemb); wts[cls_id]=0; wts[sep_id]*=sep_w; wts[dot]*=sep_w\n    Sw=(Wemb*wts[:,None]).T@Wemb\n    u=Wemb[cls_id]+Pemb[0]\n    z=np.linalg.solve(N.T@Sw@N+1e-4*np.eye(N.shape[1]), N.T@u); g=N@z\n    r=g/gam; return r/(r@eC)\nfor sw in [1,10,100]:\n    r=solve(sw)\n    used=np.where(freq>0)[0]; used=used[used!=cls_id]\n    vals=np.concatenate([rx(r,used,p) for p in [1,5,20,60]])\n    print(sw,'used mean abs',np.abs(vals).mean().round(4),'max',np.abs(vals).max().round(4),\n          'SEP',np.abs(rx(r,np.array([sep_id]*5),np.array([10,20,30,50,63]))).max().round(4),\n          'dot',np.abs(rx(r,np.array([dot]*5),np.array([10,20,30,50,63]))).max().round(4))\n# exact constraint version: add SEP and '.' word embeddings as exact constraints\nr=solve(1, extra=[Wemb[sep_id],Wemb[dot]])\nvals=np.concatenate([rx(r,used,p) for p in [1,5,20,60]])\nprint('exact sep/dot: used mean abs',np.abs(vals).mean().round(4),'max',np.abs(vals).max().round(4),\n      'SEP',np.abs(rx(r,np.array([sep_id]*5),np.array([10,20,30,50,63]))).max(), 'pos 100-160 check', np.abs(rx(r,np.array([V['the']]*3),np.array([100,150,160]))).round(4))\nnp.save('r0.npy',r)\n# typical magnitudes of q_CLS vs q_i in layer0\nWq=base('bert.encoder.layer.0.attention.self.query.weight').astype(np.float64); bq=base('bert.encoder.layer.0.attention.self.query.bias').astype(np.float64)\nqC=Wq@eC+bq\nv=Wemb[used[:500]]+Pemb[5]+Temb[0]; m=v.mean(1,keepdims=True); s=v.std(1,keepdims=True); X=gam*(v-m)/s+bet\nqi=X@Wq.T+bq\nprint('per-head |q_CLS|', np.linalg.norm(qC.reshape(12,64),axis=1).round(2))\nprint('per-head mean |q_i|', np.linalg.norm(qi.reshape(-1,12,64),axis=2).mean(0).round(2))"
}
```

