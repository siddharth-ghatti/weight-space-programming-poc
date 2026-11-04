## reasoning summary

The 768-unit head sits about 0.4% under full KRR, so I'm wondering whether the pooling layer's FFN, with its m units, could add the missing kernel capacity.

I'm weighing whether the pooling layer's FFN could supply the missing kernel capacity. The idea is to fit the FFN path and the pooler's kernel features jointly, first fitting the FFN weights by ridge regression and then fitting the readout on top. Alternatively, the FFN could carry the whole kernel machine with GELU units centered on the training points, leaving the pooler to decode the class scores.

I'm checking whether GELU units centered on all the training points can act as a kernel machine, since GELU behaves like a ReLU-style kernel in its positive range. I'll test the 5-fold accuracy of those features with plain ridge, and also check whether the FFN can read the normalized CLS input directly.

I'm working out how the FFN's intermediate weights can encode the kernel features: the first-layer rows scale each training center by the standardization, with a bias that offsets the mean. Then I'm checking whether the LayerNorm after the FFN leaves the normalized CLS vector essentially unchanged when the FFN is zeroed, so the features can be computed offline from the read vector.

I'm trying to resolve a conflict in the reserved dimensions: if the carry heads' writes are kept, those dimensions hold leftover scores that mix with the FFN's output. My plan is to remove the carry heads' writes, so the reserved dims stay zero through the network, though LayerNorm would still map that zero vector to a nonzero value unless its scale and shift are set to handle it.

I'm working out what the reserved dimensions look like once the carry heads are removed: with their output weights zeroed, those dimensions are zero before LayerNorm, and LayerNorm still turns that zero into a nonzero value, so the normalization stats shift.

I'm working through the reserved dimensions once more: with the carry heads' writes and the attention output on those dims both zero, LayerNorm maps that zero vector to a constant nonzero value, so the normalization stats depend on a single shared offset. From there I'm trying to recover the per-sample mean and scale from the read vector, since those two quantities determine the rest.

I'm checking whether a simpler proxy will do: computing the FFN features from the read vector x instead of the exact u, since the per-sample difference between them should be small. The worry is that this mismatch could flip predictions on near-tie samples, so I'd still need a final read of the logits to verify the fit.

I'm weighing whether the FFN's 3072 units could act as a full kernel machine, which would essentially match full KRR and let the pooler just decode the class scores.

I'm settling on m=3072 for every layer, since FFN pruning is off the table, which makes the pooling layer's FFN effectively free capacity. Shallow heads are out because the pooler only sees one vector, so the plan is the full stack with pooling at layer 11, then trimming layers to see how small it can get.

I'm weighing whether dropping a layer is worth it, since the 768-unit head alone doesn't reach full KRR accuracy. To test which layer to remove, I'd read out the edited stack's mean features on the 1024 subset and compare cross-validated and leave-one-out accuracy against the base model. Layers 5 and 7 show the smallest relative change in mean space, so they're the first candidates to check.

I'm considering dropping layers 5 and 7 together, since they show the smallest relative change, and if that holds up near the base accuracy I'd end up with an 8-layer stack around 81M parameters. If the paired drop fails, I'll test each layer on its own. I also want to see how accuracy changes by removing only layer 7, reading the pooled features with the pooling layer kept so the final CLS is available.

I'm considering which layers to drop, and I want to see how the depth profile looks after removing layers 5 and 7. That means reading out the hidden state after layer 8 with those two skipped, then continuing the rest of the stack to check how much depth information remains.

I'm considering a layout where the pooling layer keeps the early layers and skips a couple of middle ones, so I can see how the read-out quality holds up. The idea is to order the layer list so that each prefix is a candidate depth, letting one run compare several configurations at once.

I'm weighing how to lay out the experiment passes: ordering the layer list so each prefix is a candidate depth lets one read compare several configurations. I'm also noting that the pooling layer only gets installed in the final reads, so the exploration passes don't need it, and the remaining budget covers the final reads and a verification pass.

I'm thinking about whether the pooling layer's final CLS could be computed in numpy from the read mean and CLS vectors at the pre-pool depth, but reimplementing the layer seems more fragile than just reading the output. So the plan is to install the pooling layer and structure, then read the final CLS for all 1496 samples in two passes.

I'm going to skip the numpy reimplementation of the FFN and head, since that's fragile. Instead I'd set the pooler weights from the final CLS values I read out, then read the pooled outputs again to fit the classifier.

I'm checking whether I can fit the head on read values without simulating the encoder. Computing the pooler's tanh features from the read pooler input seems acceptable, but the FFN kernel is closer to encoder computation. The catch is that I'd need the 3072 per-sample activations to fit the output weights, and I can't read those directly.

I'm weighing a pooler-only head with 768 units fit on the read final CLS values, which costs roughly 0.4% accuracy against full KRR, a real loss at these margins. To close that gap I'm considering better center selection for the units, such as pivoted Cholesky or k-means centers, and possibly mixing in some near-linear units alongside the kernel ones.

I'm splitting the head into 128 linear PCA units and 640 kernel units, and I'll test center-selection methods to see whether that closes the gap.



With 768 units the honest 5-fold accuracy is about 97.8%, roughly 0.4 points below full kernel ridge. Next I'll try smarter centers: pivoted-Cholesky selection, k-means centroids, and a mix of near-linear and kernel units.



## tool_use: run_python

```json
{
 "code": "import numpy as np, sys; sys.path.insert(0,str(WORK))\nexec(open(WORK/'lib.py').read())\nF = np.load(WORK/'base_feats.npz'); y=F['y']\ndef LNr(X): X=X-X.mean(1,keepdims=True); return X/X.std(1,keepdims=True)\nX = LNr(5*F['mean_hidden_L9']+F['cls_hidden_L9'])[:,KEEP]\ndef pivchol(Z, kfun, m):\n    n=len(Z); d=np.array([kfun(Z[i:i+1],Z[i:i+1])[0,0] for i in range(n)]); L=np.zeros((m,n)); piv=[]\n    for j in range(m):\n        i=int(np.argmax(d)); piv.append(i)\n        kc=kfun(Z,Z[i:i+1])[:,0]; l=(kc-L[:j].T@L[:j,i])/np.sqrt(d[i]); L[j]=l; d=d-l**2; d[i]=-np.inf\n    return np.array(piv)\ndef kmeans(Z,k,it=15,seed=0):\n    rng=np.random.default_rng(seed); C=Z[rng.choice(len(Z),k,replace=False)]\n    for _ in range(it):\n        a=np.argmax(Z@C.T-0.5*(C**2).sum(1)[None],1)\n        for j in range(k):\n            if (a==j).any(): C[j]=Z[a==j].mean(0)\n    return C\ndef sor_cv(X,y,nunits=768,gamma=0.4,lams=(1e-3,1e-2),seeds=(0,1,2),sel='random'):\n    n=len(y); res={l:0 for l in lams}\n    f=lambda A,B: np.exp(2*gamma*(A@B.T-1))\n    for s in seeds:\n        for te in folds(n,5,s,y):\n            tr=np.setdiff1d(np.arange(n),te)\n            mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Z=(X-mu)/sd/np.sqrt(X.shape[1])\n            rng=np.random.default_rng(s)\n            if sel=='random': Cc=Z[rng.choice(tr,nunits,replace=False)]\n            elif sel=='piv': Cc=Z[tr][pivchol(Z[tr],f,nunits)]\n            elif sel=='kmeans': Cc=kmeans(Z[tr],nunits,seed=s)\n            H=f(Z,Cc); Kcc=f(Cc,Cc)\n            T=np.eye(8)[y[tr]]-1/8; Ht=H[tr]; G=Ht.T@Ht; R=Ht.T@T\n            for l in lams:\n                B=np.linalg.solve(G+l*Kcc+1e-8*np.trace(G)/len(G)*np.eye(len(G)),R); res[l]+=((H[te]@B).argmax(1)==y[te]).sum()\n    return {l:round(v/(n*len(seeds)),4) for l,v in res.items()}\nprint('random',sor_cv(X,y))\nprint('piv',sor_cv(X,y,sel='piv'))\nprint('kmeans',sor_cv(X,y,sel='kmeans'))\nprint('random 1196(all tr)',sor_cv(X,y,nunits=1196))\n"
}
```

