## reasoning summary

I'm checking the unseen tokens: their outputs come out small compared with the seen median, though the larger ones match sensible readings like canceled leading to term_cancel and purposely to exclusion. The units here are the a-scaled ones, since W2 includes the a=0.3 factor, so I need to compare the seen median of 0.61 on that same scale.

I'm weighing whether to smooth scores for unseen tokens using embedding similarity to seen ones, since the current model gives them roughly zero. The detectors already spread a little signal to similar tokens, so a full rebuild per fold may not be worth it for a small expected gain.

I'm weighing whether a bigram path could help, since the CV gain looks like roughly 0.3 to 0.7 percent, but any bug in the previous-token head could make spurious bigram matches cost more than that. I can only check the components analytically, so I'm leaning toward skipping it. I'm also reconsidering whether the evaluation harness's tokenization could differ from the training setup in a way that matters.

Checking that the [PAD] and [SEP] outputs hold at high positions up to 128, and that tokens near position 100 to 128 behave the same way, so the robustness claim holds across the full length. Then I'll look at float precision in the detector pre-activations.

I'm checking that the [PAD] and [SEP] outputs stay near zero at high positions, and that the seen tokens at positions 80 to 127 still fit as well as the early ones.



Unseen tokens barely move the output (99.9% under 0.2 norm versus 0.61 for seen ones), and the larger ones point sensibly. Next I'm testing robustness at positions outside the fit grid, including special tokens and seen tokens at positions 80–127.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom lib import *\nem = Emb(base)\nL='bert.encoder.layer.%d.'\nWi=W(L%0+'intermediate.dense.weight'); bi=W(L%0+'intermediate.dense.bias'); W2=W(L%0+'output.dense.weight')\nK=np.load('K.npy'); S=np.load('S_final.npy')\ndef outs(tok, pos):\n    H=em.h0(np.array(tok), pos)\n    return (gelu(H@Wi.T+bi)@W2.T)[:,DF]\nfor p in [1,64,80,100,127,200,511]:\n    o=outs([0,102,100],p); print('specials pos',p, np.abs(o).max())\nP_,y,ids,X=load_data(practice,tokenize,LABELS)\ntot=np.asarray(X.sum(0)).ravel()\ntop=K[np.argsort(-tot[K])[:500]]\nfor p in [1,10,40,63,80,100,127]:\n    o=outs(top,p); t=0.3*(S[top]-S[top].mean(1,keepdims=True))\n    rel=np.linalg.norm(o-t,axis=1)/(np.linalg.norm(t,axis=1)+1e-6)\n    cos=(o*t).sum(1)/(np.linalg.norm(o,axis=1)*np.linalg.norm(t,axis=1)+1e-9)\n    print(f'pos {p}: rel err median {np.median(rel):.3f} 95% {np.percentile(rel,95):.3f}  cos 5% {np.percentile(cos,5):.3f}')"
}
```

