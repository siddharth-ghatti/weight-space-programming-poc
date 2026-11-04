## reasoning summary

Something looks off: the full-FFN-versus-base comparison reads 0.000 at every layer, even at layers 1 and up where the input should already be drifted. Either the exposures aren't capturing drift, or the activations I'm reading were produced without the pruned layers upstream, which would mean passes 11 through 18 ran against the base model.

The FFN edits are fit on base inputs rather than drifted ones, since the full-FFN-vs-base gap is zero at every layer. The current edits were fit on all 64 sentences, so their numbers are in-sample, while the new fit uses only 32, so the comparison isn't like-for-like. With no drift present, alpha has no effect, and the drift correction still needs to be added.

Next I'm weighing how much accumulated drift a sequential fit would actually remove, since the current per-layer out-of-sample error is about 0.15 to 0.19 and compounds over nine layers. I can't simulate the drifted case without the drifted exposures, so I'm bounding it instead: more fitting data should pull the out-of-sample error toward the in-sample range of roughly 0.11 to 0.16.

Next I'm comparing pruning widths for the final choice: sequential drift-corrected at m=1024 (about 63.8M parameters, if accuracy holds) against non-sequential at m=1536 (71.7M) and m=2048 (79.6M), checking post-LN relative error at layers 0, 4, and 8 on the 32-sentence out-of-sample split.

Next I'm comparing non-sequential fits at m=1536 against the m=1024 sequential option, to see whether the wider pruning already recovers base accuracy and gives a safe fallback before I spend passes on the drift-corrected version.

Next I'm checking which FFN units actually matter: GELU units that are rarely active but strong when they fire might be worth keeping over ones with high raw energy. I'll run an offline comparison of selection methods across a few widths, measuring post-LN error on the 32-sentence held-out split.



The unpruned upstream means the current pruning is non-sequential, with drift reaching 0.44 by layer 9. Before spending nine new exposure passes on a sequential refit, I'll compare unit-selection methods offline against the base model.



## tool_use: run_python

```json
{
 "code": "import sys, json, time; sys.path.insert(0, str(WORK))\nfrom lib3 import *\nfrom lib import KEEP\nmsk20=np.load('pass_20/attention_mask.npy').astype(bool)\ndef LN(u,g,b):\n    mu=u.mean(1,keepdims=True); sd=np.sqrt(u.var(1,keepdims=True)+1e-12); return (u-mu)/sd*g+b\ndef rel(a,b): return np.linalg.norm(a-b)/np.linalg.norm(b-b.mean(0))\nsid=np.repeat(np.arange(64)[:,None],63,1)[msk20]\ndef greedy(Ac, Fc, m, lam=1.0):\n    # forward selection maximizing explained var of Fc by columns of Ac (ridge-ish via lam on diag)\n    G=Ac.T@Ac; C=Ac.T@Fc; n=G.shape[0]\n    R=G.copy(); Cr=C.copy(); d=np.diag(R).copy()+lam; sel=[]; avail=np.ones(n,bool)\n    for it in range(m):\n        gain=(Cr**2).sum(1)/np.maximum(d,1e-9); gain[~avail]=-1\n        j=int(np.argmax(gain)); sel.append(j); avail[j]=False\n        r=R[:,j].copy(); dj=d[j]\n        Cr-=np.outer(r/dj, Cr[j]); R-=np.outer(r, r)/dj; d=np.diag(R).copy()+lam\n    return np.sort(np.array(sel))\nfor l in [2,6]:\n    X,A,F,W2,bb2=layer_acts(l, base); P=f'bert.encoder.layer.{l}.'\n    g=base(P+'output.LayerNorm.weight').astype(np.float64); bt=base(P+'output.LayerNorm.bias').astype(np.float64)\n    Hb=LN(X+F,g,bt)\n    for split in [0,1]:\n        te=(sid>=32) if split==0 else (sid<32); tr=~te\n        Ac=A[tr]-A[tr].mean(0); Fcen=F[tr]-F[tr].mean(0)\n        en=A[tr].var(0)*(W2**2).sum(0)\n        t0=time.time(); sg=greedy(Ac@np.diag(np.sqrt((W2**2).sum(0)))*0+Ac, Fcen, 1536, lam=1e-3*np.trace(Ac.T@Ac)/3072); tg=time.time()-t0\n        for m in [768,1024,1536]:\n            for nm,keep in [('energy',np.sort(np.argsort(-en)[:m])),('greedy',None)]:\n                if nm=='greedy':\n                    keep=greedy(Ac, Fcen, m, lam=1e-3*np.trace(Ac.T@Ac)/3072) if m!=1536 else sg\n                As=A[:,keep]; mu=As[tr].mean(0); Acc=As[tr]-mu; Fm=F[tr].mean(0); W0=W2[:,keep].T\n                r=[]\n                for lam in [30,100,300]:\n                    Wn=np.linalg.solve(Acc.T@Acc+lam*np.eye(m), Acc.T@(F[tr]-Fm)+lam*W0); b=Fm-mu@Wn\n                    Fn=As@Wn+b; r.append(rel(LN(X+Fn,g,bt)[te][:,KEEP],Hb[te][:,KEEP]))\n                print(l,split,m,nm,' '.join('%.3f'%v for v in r))\n        print('greedy time',tg)"
}
```

