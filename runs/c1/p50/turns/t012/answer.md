## reasoning summary

I'm weighing whether RBF kernels help, since L0+L1 RBF already reaches 97.4% and nonlinearity seems worth about a point. I'm also considering whether the pooler's 768 tanh units plus the classifier could approximate kernel ridge through random features. Next I want to check the errors for confusable cases, and then think about which features attention pooling should compute.

I'm considering an alternative with a single attention pooling layer over the embedding outputs, followed by an FFN, pooler, and classifier. That layout could act as a two-hidden-layer MLP on the pooled features, and I'm comparing its parameter cost against the two-layer version.

I'm weighing whether a token-level nonlinearity before pooling would beat the linear mean: per-token features followed by mean pooling act like a nonlinear bag-of-words, and the original layer 0 with its FFN gets the mean to about 95.9% linear, while L0+L1 RBF reaches 97.4%. I'd like a target of at least 97.8% with k ≤ 2, though I don't yet know if that's reachable, so I'll look at the error analysis first to see whether the errors are hopeless.



A nonlinear RBF kernel ridge head lifts layer 0+1 accuracy to 97.4%, versus 96.4% linear, and the pooler's tanh units can approximate that nonlinearity cheaply. Next I'll run error analysis to see which classes get confused and whether those errors look recoverable.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\nd=load_base(); y=labels(); P=[json.loads(l) for l in open('practice.jsonl')]\nM=lambda l: d[f'mean_hidden_L{l}'][:,KEEP]\nX=np.concatenate([M(2),M(6),M(10),d['cls_hidden_L12'][:,KEEP]],1)\nr=cv_acc(X,y,lams=(10,),ret_scores=True,seeds=(0,)); S=r[('S',10)]; p=S.argmax(1)\nX0=M(0); r0=cv_acc(X0,y,lams=(3,),ret_scores=True); p0=r0[('S',3)].argmax(1)\ncm=np.zeros((8,8),int)\nfor a,b in zip(y,p0): cm[a,b]+=1\nprint(cm)\nerr=np.where(p!=y)[0]; print(len(err), 'L0 errs', (p0!=y).sum(), 'both', ((p0!=y)&(p!=y)).sum())\nfor i in err[:40]:\n    print(LBL[y[i]][:10],'->',LBL[p[i]][:10],'|',P[i]['register'],'|',P[i]['text'][:150])"
}
```

