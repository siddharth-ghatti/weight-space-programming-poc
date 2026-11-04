## reasoning summary

I'm working through layer 0 head by head, starting with how far each head's attention pattern diverges from uniform, measured as KL. Next I'm comparing each head's actual output against what it would produce with uniform attention, and I'm checking how large that error is relative to the residual stream.

I'm now comparing the dropped-head error against the replacement error and then choosing the head whose uniform replacement changes the output least.



## message

Now, for each layer-0 head: how much its output changes if its attention is forced to be uniform (exact for a carry head), measured relative to the size of the residual stream.

## tool_use: run_python

```json
{
 "code": "import numpy as np, json\nA=np.load('pass_23/attention_L1.npy').astype(np.float64); H0=np.load('pass_23/token_hidden_L0.npy').astype(np.float64)\nH1b=np.load('pass_23/token_hidden_L1.npy'); msk=np.load('pass_23/attention_mask.npy').astype(bool)\nP='bert.encoder.layer.0.attention.'\nWv=W(P+'self.value.weight').astype(np.float64); bv=W(P+'self.value.bias').astype(np.float64)\nWo=W(P+'output.dense.weight').astype(np.float64); bo=W(P+'output.dense.bias').astype(np.float64)\nV=H0@Wv.T+bv  # n,T,768\nn,T,_=H0.shape\ntot=np.zeros((n,T,768)); outs=[]\nfor h in range(12):\n    sl=slice(64*h,64*h+64)\n    ctx=np.einsum('nts,nsd->ntd',A[:,h],V[:,:,sl])\n    o=ctx@Wo[:,sl].T; outs.append(o); tot+=o\ntot+=bo\nU=H0+tot  # pre-LN\nden=np.sqrt(((U[msk]-U[msk].mean(1,keepdims=True))**2).sum(1)).mean()\nL=msk.sum(1)\nfor h in range(12):\n    sl=slice(64*h,64*h+64)\n    mV=(V[:,:,sl]*msk[:,:,None]).sum(1)/L[:,None]  # uniform mean\n    ou=np.repeat((mV@Wo[:,sl].T)[:,None,:],T,1)\n    err_unif=np.linalg.norm((outs[h]-ou)[msk],axis=1).mean()/den\n    err_drop=np.linalg.norm(outs[h][msk],axis=1).mean()/den\n    # entropy ratio\n    a=A[:,h]; ent=-(a*np.log(a+1e-12)).sum(-1); entu=np.log(L)[:,None]\n    er=(ent/entu)[msk].mean()\n    print(h,'unif-replace err %.4f  drop err %.4f  entropy/uniform %.3f'%(err_unif,err_drop,er))"
}
```

