## reasoning summary

I'm testing an RBF kernel ridge on the CLS and mean features from layer 11, which reaches 0.983 against 0.975 for the linear probe. Wider bandwidths and a smaller regularizer are next to try. I'm also asking whether the final layer's 3072-neuron GELU FFN on the CLS token could reproduce that kind of gain.

I'm checking whether random GELU features on the 768-d superposition x can match the RBF kernel's roughly 0.98, and whether the final layer's FFN could carry that readout. If it works, I'd write the ridge solution into the residual through the second FFN projection, then let the layer norm rescale the result.

Now I'm weighing whether the FFN hidden activations can be read out at all. The tool exposes CLS and mean states, logits, pooled output and attention, but not the intermediate neurons, so I'd need to route the GELU activations through an output I can observe, such as the pooler, before I could fit anything against them.

I'm looking at whether the pooler could serve as the random-feature layer: a tanh over a random projection of the residual, with a linear classifier fit by ridge on the pooled outputs read through the tool. That would be 768 features rather than 3072, so the kernel approximation may be weaker, and I want to check that offline before committing to it.

I'm weighing whether computing the pooler's random features in numpy would just be simulating the model, which the rules forbid. I'd rather stick to generic kernel tests and then confirm the approach with actual tool reads, so I'm sketching the design: LN(cls11 + M·mean11) feeding random pooler features, read out and fit with ridge.

I'm checking the RBF kernel on a label-free mix of the CLS and mean states, x = c11 + 2·m11, since fitting M on labels would make the pooled-feature CV optimistic. I'll verify the kernel offline and then confirm with real reads. Padding the random-feature layer with the FFN's random projection is the next idea I'm sketching.

I'm going to check the RBF kernel on a label-free mix of the CLS and mean states, x = c11 + 2·m11, and also on a layer-normalized version of x. I'll run a quick offline test with random tanh features on the standardized features to get a sense of whether 768 of them are enough.



An RBF kernel on the CLS and mean features reaches about 98.3% against 97.5% for linear, so I'll approximate it with random tanh features and fit a ridge classifier in closed form. I'm first testing bandwidths offline, using the label-free superposition cls11 + 2·mean11.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nD=np.load('base_feats.npz'); y=D['y']; folds=np.load('folds.npy')\nc11=D['cls11'].astype(np.float64); m11=D['mean11'].astype(np.float64)\nT=np.eye(8)[y]\ndef krr_cv(K,lam):\n    pred=np.zeros(len(y),int)\n    for f in range(5):\n        tr=folds!=f\n        a=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()),T[tr]-T[tr].mean(0))\n        pred[~tr]=np.argmax(K[np.ix_(~tr,tr)]@a,1)\n    return (pred==y).mean()\ndef rbf_report(name,X):\n    X=(X-X.mean(0))/(X.std(0)+1e-6)\n    sq=(X**2).sum(1); D2=np.maximum(sq[:,None]+sq[None,:]-2*X@X.T,0); med=np.median(D2)\n    for gmul in [2,4,8]:\n        K=np.exp(-D2/(gmul*med)); print(name,'rbf',gmul,[round(krr_cv(K,l),4) for l in [0.001,0.01,0.03]],flush=True)\nrbf_report('concat',np.hstack([c11,m11]))\nx=c11+2*m11\nrbf_report('sum2',x)\nxc=x-x.mean(1,keepdims=True); ln=xc/xc.std(1,keepdims=True)\nrbf_report('LN(sum2)',ln)\nrbf_report('mean11',m11)"
}
```

