## reasoning summary

I'm comparing layer accuracies on the full 1496 set, and L9 comes out clearly ahead at 98.37%. L4 scores unexpectedly high at 97.97%, so I'm wondering whether its lexical features are doing the work. The standard error is around 0.35%, but since the depth candidates are paired, the differences should be more reliable than that suggests.

I'm weighing whether dropping layer 8 is worth it, and at 97.9 it's too close to 97.8 to bother. Instead I'm considering whether a linear score from M4 carried through the residual channel could stand in for the last layers, though that would mean repurposing one of layer 4's heads.

I'm checking whether the old carry head in layer 6 (head 11) can be turned back on to carry linear scores from the layer 5 output, since its output columns already only feed the residual channel.

I'm checking whether the free carry heads in layers 2 and 6 could carry linear scores from earlier hidden states, which would give the pooling layer more to work with than the last layer's state alone. Uniform attention makes their output linear in those states, so averaging over tokens still keeps it linear.

I'm checking whether the carried linear terms from M0 and M6 can be fit jointly, since each one gets scaled by a different per-sample factor as it passes through the layer norms.

I'm going to test whether a linear term from an earlier layer, M2, M4, or M6, adds predictive value on top of the kernel features from M7, M8, or M9. I'll run a joint fit on the carried scores, normalizing each one, since I only have edited features for M4 through M8 so far.



Dropping layer 8 would leave CV near 97.9%, just above the 97.8% bar, so I'm testing whether linear class-score features from the old carry head in layer 6 can push a shallower last layer to about 98.3%.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nfrom lib7 import *\nimport lib7\nF={k:v.astype(np.float64) for k,v in np.load('feats_seq1024_depth.npz').items()}\ny=labels_for([f'p{i:04d}' for i in range(1496)], practice, LABELS)\ndef cv2(last, extra, c0s=None, seeds=(0,1,2), lam=0.02):\n    M=F[f'M{last}']; C=F[f'C{last}']\n    res=[]\n    for sd in seeds:\n        f=folds(y,10,sd); Y=np.eye(8)[y]; pred=np.zeros(len(y),int)\n        for i in range(10):\n            tr=f!=i; te=~tr; mu=M[tr].mean(0)\n            Z=pool_z(M,C,10.0,mu); Zt=Z[tr]; Wf=anchor_dirs(Zt); Kc=gelu(Zt@Wf.T); Kc=.5*(Kc+Kc.T)\n            piv=pivchol(Kc,1008); Wd=Wf[piv]; Kaa=Kc[np.ix_(piv,piv)]; Phi=feats(Z,Wd)\n            Lin=np.concatenate([lin_feats(F[e],tr) for e in extra],1) if extra else np.zeros((len(y),0))\n            coef,b=fit(Phi[tr],Lin[tr],Y[tr],Kaa,lam,0.5/max(1,len(extra)))\n            pred[te]=(np.concatenate([Phi[te],Lin[te]],1)@coef+b).argmax(1)\n        res.append((pred==y).mean())\n    return np.mean(res), np.round(res,4)\nfor last in [9,8,7]:\n    for extra in [['M0'],['M0','M4'],['M0','M6'],['M0','M5'],['M0','M4','M6']]:\n        if any(int(e[1:])>=last for e in extra): continue\n        print(last, extra, cv2(last,extra), flush=True)\n"
}
```

