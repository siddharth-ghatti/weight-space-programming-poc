## reasoning summary

I'm counting the exposure passes: 64 expo_ids, and I want the token totals for passes 10 through 18. Separately, tok64 with its different ids comes to 1901 tokens across all 13 layers. There's also a drift issue: layers beyond 0 were exposed through the already-pruned earlier layers, so I need the base hidden state at layer l+1 for those same expo_ids to correct for it.

I'm weighing whether the drift can be fixed without new exposures: I could take base hidden states for the same expo_ids in one pass, and correct layer l's fit against those targets. But changing layer l makes the drifted inputs for layer l+1 stale, so a proper sequential redo would need about 10 fresh passes, leaving 21 for evaluation and final building.

I'm weighing whether to check how well a pruned fit holds up before spending passes. For layer 0, where the input is pristine, I can test offline by fitting on half the sentences and measuring reconstruction error on the rest across several values of m.

I'm planning to check the drift-corrected fit against held-out tokens by measuring the relative error on both the FFN output and the pre-LN residual stream. I also want to try greedy forward selection and compare training on 16 versus 32 sentences to see whether more data helps.



The exposure passes will supply FFN inputs for 64 sentences per layer. Before running them, I'm measuring offline how well a pruned FFN reconstructs the full one on held-out sentences, varying m, the selection method, and the amount of fit data.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0, str(WORK))\nfrom lib3 import *\nimport time\ndef acts_with_sent(l):\n    emap=json.load(open('expo_map.json')); k=emap[str(l)]\n    Zp=np.load(f'pass_{k}/token_hidden_L{l+1}.npy'); msk=np.load(f'pass_{k}/attention_mask.npy').astype(bool)\n    sid=np.repeat(np.arange(msk.shape[0])[:,None],msk.shape[1],1)[msk]\n    return sid\nfor l in [0,4,8]:\n    X,A,F,W2,bb2=layer_acts(l, base); sid=acts_with_sent(l)\n    tr=sid<32; te=~tr\n    print('layer',l,'tokens',len(sid),'train',tr.sum())\n    Fm=F[tr].mean(0); denF=np.linalg.norm(F[te]-Fm); denXF=np.linalg.norm((X+F)[te]-(X+F)[tr].mean(0))\n    en=A[tr].var(0)*(W2**2).sum(0)\n    for m in [512,1024,1536]:\n        keep=np.sort(np.argsort(-en)[:m])\n        # no refit, mean compensation\n        rem=np.setdiff1d(np.arange(3072),keep)\n        F0=A[:,keep]@W2[:,keep].T+bb2+A[tr][:,rem].mean(0)@W2[:,rem].T\n        out=[('meancomp',np.linalg.norm(F0[te]-F[te])/denF)]\n        for lam in [10,100,1000]:\n            As=A[:,keep]; mu=As[tr].mean(0); Ac=As[tr]-mu; Fc=F[tr]-Fm; W0=W2[:,keep].T\n            Wn=np.linalg.solve(Ac.T@Ac+lam*np.eye(m), Ac.T@Fc+lam*W0); b=Fm-mu@Wn\n            Fh=As@Wn+b\n            out.append((lam, np.linalg.norm(Fh[te]-F[te])/denF))\n        print(' m',m, ' '.join(f'{a}:{b:.3f}' for a,b in out), ' (relXF meancomp %.3f)'%(np.linalg.norm(F0[te]-F[te])/denXF))"
}
```

