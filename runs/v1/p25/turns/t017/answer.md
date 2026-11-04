## reasoning summary

I'm now setting up a token-level linear map from the layer-11 hidden states to the layer-0 states, fitted on about 4000 tokens from the 128 sentences, and then applying it to all 1496 sentences to train a classifier on both representations. I'll also try fitting the same map from layer 6 to layer 0 and start on a held-out token-level R² by splitting those 128 sentences.

I'm splitting the 128 sentences into two halves to measure held-out token-level R².



Token states are in for 128 sentences, so I'm fitting a token-level ridge map from h11 to h0 and checking its held-out R² before testing whether it lifts the M0 classifier.



## tool_use: run_python

```json
{
 "code": "import json\ndef load_tok(d):\n    am=np.load(d+'attention_mask.npy').astype(bool)\n    H={L:np.load(d+f'token_hidden_L{L}.npy') for L in [0,6,11]}\n    return am,H\ntoks={L:[] for L in [0,6,11]}; sid=[]\nfor k,d in enumerate(['pass_4/','pass_5/']):\n    am,H=load_tok(d)\n    for i in range(am.shape[0]):\n        for L in [0,6,11]: toks[L].append(H[L][i][am[i]])\n        sid+= [k*64+i]*am[i].sum()\nT={L:np.concatenate(toks[L]) for L in [0,6,11]}; sid=np.array(sid)\nnp.savez('tok128.npz', sid=sid, **{f'T{L}':T[L] for L in T})\nprint(T[0].shape)\n# held-out R2 token-level\ntr=sid%2==0; te=~tr\nfor src in [11,6]:\n    X=T[src]; Yt=T[0]; mx=X[tr].mean(0); my=Yt[tr].mean(0)\n    for lam in [1,10,100,1000]:\n        R=np.linalg.solve((X[tr]-mx).T@(X[tr]-mx)+lam*np.eye(768),(X[tr]-mx).T@(Yt[tr]-my))\n        pr=(X[te]-mx)@R+my\n        print(src,lam,'tokR2',round(1-((pr-Yt[te])**2).sum()/((Yt[te]-my)**2).sum(),3))"
}
```

