## reasoning summary

The fusion results sit around 97.3 to 97.5 percent, and the joint ridge across all mean layers reached 98.0, which is the best so far. Adding a third layer to the fusion, as with m0, m6, and m11, didn't improve on the two-layer version, so late fusion seems to top out there. I'm now weighing whether carrying more information forward, through the joint approach, could push past that ceiling.

Joint ridge over m2 and m11 only reaches 97.6, slightly above fusion's 97.3, which may be within noise. I'm wondering whether early layers could each add their signal into the same carried directions, so that a head at layer 2 and another at layer 4 both contribute to one shared residual signal.

I'm now seeing that the joint ridge over concatenated layers splits into per-layer blocks, so each layer's head could add its contribution into the same residual coordinates. The catch is that the edited model's intermediate states drift from the base ones, so I'd need to fit on the edited features at each layer rather than the base ones.

I'm working out how to carry the joint ridge's per-layer blocks into several heads at once, each adding its piece into the same residual coordinates. The ablation shifts later layers' states, so the heads at layers 4, 6, 8, and 10 would have to be fit on the edited features rather than the base ones. Installing a head at layer 4 also changes what layers 5 onward see, which complicates the fit.

I'm seeing that the joint ridge over several layers' means decomposes into per-layer blocks, so a head at each layer could add its piece into the same residual coordinates. Since installing those heads only writes to coordinates no other layer reads, I can prepare the model first, fit everything jointly, and then install the blocks without disturbing the other hidden states.

I'm checking whether the carried heads can all be installed without disturbing the hidden states they read. Installing them into sacrificed heads only writes to the carried coordinates, but those coordinates feed the layer norms downstream, so there's a small shift to account for, roughly a percent or so in variance.

I'm checking whether the carried signal would shift the layer norm statistics, and it looks like the effect is small: about a percent in variance and a negligible shift in the mean. I could cut that further by making the carried logits zero-sum across classes, which the ridge weights nearly already are, and dropping the intercept to match, while keeping the gain small.

I'm checking that the learned gains on the carried residual stay small, since larger ones would disturb the layer norm statistics. The final readout can still decode a small signal, so I'm weighing the late ridge on the final model's features against the carried logits, with the fusion weight chosen from out-of-fold simulation. The joint ridge over all mean layers is what reached 98.0, and I'm comparing that against this two-stage version.

I'm checking whether a two-stage setup, with carried logits from early and mid layers plus a late ridge on the final features, could beat the 97.3 to 97.5 fusion, and I'll simulate it out-of-fold to see if it reaches roughly 98. Carry heads can only sit at encoder layers 2 through 10, since each one has to read that layer's hidden state, and the layer-11 aggregator covers the rest.

I'm checking whether layer 0 can be carried too, by wiping at the output of layer 0's norm and using heads at layers 1 through 10. The embedding-level content in the residual would make a head at layer 0 awkward, since its contribution can't be cleanly separated from what's already there.

I'm weighing a set of simulations to compare: a joint ridge over the early and middle layers, a version that adds layer 1, and one that uses layer 0 instead, with the fully joint ridge over every layer as the upper bound to measure against.

I'm settling on a comparison: fit the joint ridge on early and middle layer features, add layer 11 from the edited late readout, and see whether that beats the 97.3 fusion. I'll use the edited E12 non-R features for the late part since they're already available.



The two results hold: the edits didn't hurt the late features, and fusing with the layer-2 logits reaches about 97.3–97.5% CV. Next I'll simulate whether carrying the signal across several layers is worth it.



## tool_use: run_python

```json
{
 "code": "lab=np.load(WORK/'labels.npy'); N=len(lab); folds=np.load(WORK/'folds.npy'); Y=np.eye(8)[lab]\nR = [601, 222, 88, 83, 444, 506, 254, 495]; keep=np.setdiff1d(np.arange(768),R)\nM=np.load(WORK/'base_mean.npy'); E12=np.load(WORK/'ed1_cls12.npy'); a=np.load(WORK/'ed1_a.npy')\ndef oof(X,lam):\n    o=np.zeros((N,8))\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Z=(X[tr]-mu)/sd\n        if Z.shape[1]>Z.shape[0]:\n            K=Z@Z.T; al=np.linalg.solve(K+lam*np.eye(len(K)),Y[tr]-Y[tr].mean(0)); o[te]=((X[te]-mu)/sd)@(Z.T@al)\n        else:\n            W=np.linalg.solve(Z.T@Z+lam*np.eye(Z.shape[1]), Z.T@(Y[tr]-Y[tr].mean(0))); o[te]=((X[te]-mu)/sd)@W\n    return o\nacc=lambda S:round((S.argmax(1)==lab).mean(),4)\nlate={lam:oof(E12[:,keep],lam) for lam in [30,100]}\nsets={'2':[2],'2,4,6,8,10':[2,4,6,8,10],'2..10':list(range(2,11)),'1..10':list(range(1,11)),'0,2..10':[0]+list(range(2,11)),'2,6,10':[2,6,10],'2,4':[2,4], '2,3':[2,3]}\nfor name,ls in sets.items():\n    X=np.concatenate([M[l][:,keep] for l in ls],1)\n    for lam in [300,1000]:\n        c=oof(X,lam*len(ls)**0.5)\n        print(name, lam, 'carried', acc(c), 'fusion', [(al,acc(al*c*(a/a.mean())[:,None]+late[30])) for al in [0.75,1,1.5,2]], flush=True)\n"
}
```

