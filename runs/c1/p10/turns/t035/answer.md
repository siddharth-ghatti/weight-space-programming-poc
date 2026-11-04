## reasoning summary

On the 1024 subset, the base M9 leave-one-out score is 97.75%, well below the 98.66% from the full 1496 set, so sample size matters a lot here. Dropping layers 5 and 7 gives 96.97% after layer 8, roughly the base M5 level, and it rises to 97.66% after layer 10, close to base M9. So removing two layers costs about 0.8% at the same depth, and adding layers 9 and 10 wins most of that back. That isn't a saving, since [0,1,

I'm checking whether base M11 is actually the strongest on the 1024 subset, but with samples that small, gaps of about 0.5% are within noise. Paired comparisons on identical samples should cut that noise somewhat. Dropping two layers clearly hurts, and a single drop probably costs around 0.4%, so the current safe configuration still looks like the better choice.

I'm wondering whether the CLS token could be made to attend uniformly in layer 8, so pooling happens inside that layer rather than as a separate step. The obstacle is that the CLS state differs from sentence to sentence, so a fixed linear adjustment to the query can't cancel its attention exactly. One possibility is a large component in a direction that only CLS activates and that the key bias leaves constant across tokens.

Pooling would only work if one head in layer 8 could attend uniformly for the CLS token, and that can't be done exactly with a linear query adjustment since the CLS state varies per sentence. A dedicated head could carry the mean of layer 8's input, but that's still M8 information rather than M9. Getting M9 would seem to require a full extra attention layer after layer 8.

Pooling inside the final real layer is out, since its attention can only give M8 features. I'd need a full extra attention layer after layer 8 to reach M9. Maybe the pooling layer could be made to process the CLS state in some other way, though I haven't worked out how.

I'm weighing whether mildly pruning the FFN, dropping about a sixth of its units by a least-importance proxy without any compensation, would keep accuracy. The m=1024 experiment wiped out the features, so I'm unsure a gentler cut would hold, though it would bring the model down to roughly 87.4M.

I'm considering whether a gentle FFN pruning at around 2560 units could keep accuracy, though the proxy ranking's reliability is uncertain. Shifting to the shallow route, I want to compare the errors of the M1 kernel against the M9 kernel to see which sentences fail and whether negation drives the difference.

I'm going to look at a few of the misclassified sentences from both kernels, to see whether the errors cluster around things like negation that a shallow bigram-style feature would miss.



Dropping layers 5 and 7 cuts accuracy to about 96.97%, close to base M5, and adding 9 and 10 back gains nothing, so the safe design is [0..8] plus pooling at 95.3M. Before accepting the 13% reduction, I'm checking where the shallow model's errors differ from the deep one's, in case a pattern makes a 2-layer model fixable.



## tool_use: run_python

```json
{
 "code": "import numpy as np, sys, json; sys.path.insert(0,str(WORK))\nexec(open(WORK/'lib.py').read())\nF = np.load(WORK/'base_feats.npz'); y=F['y']; P=practice()\ndef LNr(X): X=X-X.mean(1,keepdims=True); return X/X.std(1,keepdims=True)\ndef zsall(X): mu=X.mean(0); sd=X.std(0)+1e-6; return (X-mu)/sd/np.sqrt(X.shape[1])\ndef loo_pred(Z,y,g,l):\n    sq=(Z**2).sum(1); D2=np.maximum(sq[:,None]+sq[None]-2*Z@Z.T,0); T=np.eye(8)[y]-1/8\n    K=np.exp(-g*D2); w,V=np.linalg.eigh(K); f=w/(w+l); Hd=((V**2)*f[None]).sum(1); Yh=V@(f[:,None]*(V.T@T))\n    return (Yh-Hd[:,None]*T)/(1-Hd[:,None])\nZ01=np.concatenate([zsall(F['mean_hidden_L0'][:,KEEP]),zsall(F['mean_hidden_L1'][:,KEEP])],1)/np.sqrt(2)\nS01=loo_pred(Z01,y,0.8,0.001); S9=loo_pred(zsall(LNr(5*F['mean_hidden_L9']+F['cls_hidden_L9'])[:,KEEP]),y,0.4,0.003)\ne01=np.where(S01.argmax(1)!=y)[0]; e9=np.where(S9.argmax(1)!=y)[0]\nprint(len(e01),len(e9),'both',len(set(e01)&set(e9)))\nab=[LABELS[i][:10] for i in range(8)]\nfor i in e01:\n    tag='BOTH' if i in e9 else 'shallow-only'\n    print(tag, ab[y[i]],'->',ab[S01[i].argmax()],'|',P[i]['text'][:150])\nprint('---- deep-only errors')\nfor i in sorted(set(e9)-set(e01)): print(ab[y[i]],'->',ab[S9[i].argmax()],'|',P[i]['text'][:150])\n"
}
```

