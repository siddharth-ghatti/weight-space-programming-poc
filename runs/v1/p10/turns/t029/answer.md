## reasoning summary

The correlation is already 0.81 at the first layer, so most of the fidelity loss seems to happen right at layer 0, either in x0 versus e or in the layer norm output there. Passing through the later layers barely degrades it further, dropping only from 0.81 to 0.78. Now I'm wondering whether the mean over tokens is distorting things too, since it includes the CLS and SEP tokens.

I'm checking whether fitting A on x0 would help, since x0 is what the layer-0 FFN actually reads and I only observe e. Zeroing the layer-0 attention output would change the network too much, and x0 itself isn't exposed by the read tool, so the layer-0 mismatch is hard to fix directly.

I'm weighing a trick to avoid the layer-0 mismatch: give one head in layer 0 uniform attention, so its value projection averages over all tokens. That would reproduce the sentence-mean projection exactly, and the token weighting would drop out. The cost is that this head's original function is lost, and I'd have to work out which dimensions to repurpose.

I'm weighing whether to give up one layer-0 head by making its attention uniform, since that would reproduce the sentence-mean projection exactly. The open question is which head to sacrifice, and I'd want its original function to be as close to a broad average as possible so the damage stays small. Finding that head cleanly would mean reading attention patterns, and I'm not sure I can do that without running more of the forward pass myself.

I'm now considering an alternative that leaves the attention pattern alone: keep an existing head's attention as it is and repurpose some of its value dimensions to carry the projection. That would give an attention-weighted average of the projected tokens rather than an exact match, so it's not a clean fix either.

I'm weighing whether to sacrifice a layer-0 head by forcing uniform attention, which would give the exact sentence-mean projection with no token weighting. The catch is choosing which head to lose so the damage to the other features stays small.

I'm wondering whether the mismatch comes from layer 0's attention mixing neighboring tokens into x0, or from LayerNorm renormalizing the vector, which could shift the dimensions A was fit on. The mixing itself probably doesn't hurt much, since it's still roughly bag-of-words. The more likely problem is that A was fit on raw embeddings, where some directions are dominated by position components that get reweighted once x0 is formed.

I'm now asking whether the standardized ridge fit is the real problem: by scaling each dimension, it gives low-variance directions of the sentence means extra weight, and those directions may carry different content once the layer-0 attention output is added. Fitting A without per-dimension standardization, or with stronger regularization, could reduce that noise. Since x0 sits between e and h1, a proxy closer to x0 might also help, though I'm not sure which observable one works best.

I'm testing whether the mismatch comes from how A was fit: if the standardized ridge fit gives low correlation between m0@A and m1@A, while an unstandardized fit keeps them aligned, that would explain the gap. I'll also compare the actual layer-1 D signal against m1@A and m0@A to see whether x0 sits closer to h1 or to e.



Most of the fidelity loss comes from layer 0, since correlation is already 0.81 at L1, so I'll check whether the real L1 signal tracks m1@A better than m0@A and how stable the A-fitting variants are across m0 and m1.



## tool_use: run_python

```json
{
 "code": "exec(open('helpers.py').read())\nlab=labels_all(); y=lab[:1024]\nD = np.load('D_dims.npy'); A=np.load('A_proj.npy'); nonD=np.setdiff1d(np.arange(768),D)\nCp=np.eye(8)-1/8\nm0 = load('mean',0,'pass_2'); m1b = load('mean',1,'pass_2'); m11b=load('mean',11,'pass_2')\nact1 = load('mean',1,'pass_3')[:,D]@Cp\nc0 = [np.corrcoef(act1[:,j],(m0@A@Cp)[:,j])[0,1] for j in range(8)]\nc1 = [np.corrcoef(act1[:,j],(m1b@A@Cp)[:,j])[0,1] for j in range(8)]\nprint('act L1 vs m0@A', np.round(c0,2)); print('act L1 vs m1@A', np.round(c1,2))\nprint('m0@A vs m1@A', np.round([np.corrcoef((m0@A)[:,j],(m1b@A)[:,j])[0,1] for j in range(8)],2))\n# alternative A: unstandardized ridge\ndef ridge_raw(X,y,lam):\n    mu=X.mean(0); Xc=X-mu; Y=np.eye(8)[y]\n    return np.linalg.solve(Xc.T@Xc+lam*len(y)*np.eye(X.shape[1]), Xc.T@(Y-Y.mean(0)))\nfor lam in [1e-4,1e-3,1e-2]:\n    A2 = ridge_raw(np.concatenate([m11b,m0],1), y, lam)[768:]\n    print('raw lam',lam,'m0@A2 vs m1@A2', np.round(np.mean([np.corrcoef((m0@A2)[:,j],(m1b@A2)[:,j])[0,1] for j in range(8)]),3))\nprint('sd of m0 dims: min/median/max', np.percentile(m0.std(0),[0,50,100]))\n"
}
```

