## reasoning summary

I'm checking whether the 8 ablated dimensions matter much: they're a small share of the model's input, and the readout gets refit anyway. I'm also considering leaving those dimensions unzeroed and letting the network read them.

I'm now weighing a different option: letting the network read the stash dims rather than zeroing them, but that would feed stash content in as noise, so mean-ablation is the better choice. I'm picking eight stash dimensions that don't overlap with the channel dims, then turning to head 0's value dims in layer 0 and selecting the eight with the least variance in their output contribution to mean-ablate.

I'm working out how to set the stash dims at layer 0 so their magnitude stays stable, starting with the LN gain at each stash dim. Then I plan a single diagnostic read of CLS hidden states across layers 1 to 9 for 256 texts, to compare the stash values against what I intended.

I'm estimating how much the drift in stash dims matters, and I'll check it empirically instead of guessing. I'm also considering the FFN side of layer 0: zeroing its W2 rows and W1 columns for the stash dims, with a bias correction in b1 based on the mean LN1 output at those dims, which I'd need to approximate from the data.

I'm reconstructing the remaining step: the joint ridge fit over the final-layer CLS, a mid-layer, and the layer-0 mean features, to recover the stash values I intend to write.

I'm setting up the joint ridge fit on the base features to get the stash projection A0 and offset b0, so the centered stash values come out with zero mean over the practice set. Then I'll scale the coefficients so the stash contribution lands near magnitude 0.5, and move on to building the layer-0 edits.

I'm going to skip the stash-dim gain choice for now and move on to building the layer-0 edits and ablations for layers 0 through 8. The layer-9 readout doesn't affect the diagnostic, since the hidden states at those layers come before any layer-9 edits. Then I'll run one read pass over the first 1024 texts and the remaining 472, checking drift at layers 1 through 9, and only refit if it looks bad.

I'm weighing whether to ablate at every layer or only some, and I'm settling on mean-ablation for the stash dims, with the layer-9 readout left to refit later since it doesn't affect the layer 0 to 9 hidden states I'm diagnosing.

I'm now setting up the head 0 layer-0 code: with the query zeroed, attention is uniform, and the value dims are the first 64 entries of Wv and the matching Wo columns. I'll save the stash configuration to a file once the code is written.



Ablation scores are fairly flat across dims, so removing 8 of 768 should cost little, and I'll refit the readout afterward. Next I'll compute the stash content A0 from a joint ridge on base features, then choose the 8 head-0 value dims with the least varying mean-ablation.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nF=np.load('feats.npz'); y=F['y']\nC9=F['C9'].astype(np.float64); M9=F['M9'].astype(np.float64); M0=F['M0'].astype(np.float64)\nS=np.array([654,529,216,175,145,159,531,517])\n# joint ridge\nXs=[]; mus=[]; sds=[]\nfor X in [C9,M9,M0]:\n    m=X.mean(0); s=X.std(0)+1e-6; mus.append(m); sds.append(s); Xs.append((X-m)/s)\nZ=np.concatenate(Xs,1); Y=np.eye(8)[y]-1/8\nZc=Z-Z.mean(0)\nlam=1000\na=np.linalg.solve(Zc@Zc.T+lam*np.eye(len(Z)),Y-Y.mean(0)); Wt=Zc.T@a\nW0=Wt[1536:]                      # on stdz M0\nA0=(W0/sds[2][:,None]).T          # [8,768] on raw M0\nb0=-(mus[2]/sds[2])@W0            # makes stash zero-mean\nst=M0@A0.T+b0\nprint('stash std per class', np.round(st.std(0),3), 'mean', np.round(st.mean(0),4))\nprint('stash-only acc (in-sample)', (st.argmax(1)==y).mean())\nfull=Z@Wt+ (Y.mean(0)-Z.mean(0)@Wt); print('joint in-sample acc',(full.argmax(1)==y).mean())\n# head-0 value dims to sacrifice\nL='bert.encoder.layer.0.'\nWv=base(L+'attention.self.value.weight').astype(np.float64); bv=base(L+'attention.self.value.bias').astype(np.float64)\nWo=base(L+'attention.output.dense.weight').astype(np.float64)\nv=M0@Wv[:64].T+bv[:64]            # [n,64] uniform-head values\ncost=v.var(0)*np.linalg.norm(Wo[:,:64],axis=0)**2\nJ=np.argsort(cost)[:8]; print('J',J,'cost',np.round(cost[J],4),'median',np.round(np.median(cost),4))\n# how big is head-0's output in stash dims originally / CLS embedding at stash dims\nnp.savez('stash_cfg.npz',S=S,J=J,A0=A0,b0=b0,vmean=v.mean(0))"
}
```

