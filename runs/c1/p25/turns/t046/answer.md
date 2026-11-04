## reasoning summary

I'm weighing the M9 quadratic with the linear M2 at about 97.95, against M8 near 97.5, so the jump from layer 8 to 9 is roughly 0.45 points. The kernel seems to add about that much. The start model reports CV 97.8 with a joint ridge over M2, M6, M10 and the final CLS, and gets 98.31 on test. Now I'm working out what k=10 looks like using layers 0 through 8 pooled with the kernel, and what CV that might give.

I'm checking whether k=10 with the kernel readout beats the start model's 97.8 CV, since the estimate is about 97.95. The harder question is how to implement the quadratic kernel. Explicit C² features would be 768² dimensions, so I'm considering a Nyström approximation instead, mapping inputs through GELU units in the pool slot. That only holds if the features are actually computable from the activations, and I'm not sure they are.

I'm weighing a plan that avoids simulating the forward pass in numpy: fit a kernel proxy, write the weights, then read the real final-layer hidden states and refit the linear readout on those actual activations. That keeps the closed-form fitting on real reads and stays clear of the forward-pass restriction.

I'm turning to the design of the pool layer, which will be layer 11 since it already carries the pooling attention. Its input is the layer-8 output, the mean M9 state. The question is how to write the FFN output so it yields kernel features, feeding the existing pooler and classifier.

I'm thinking about using the pooler itself as the kernel-style feature layer: its 768 tanh units act as random features over the hidden state. I could read the pooled outputs for the practice ids and fit the 8-way classifier by ridge with CV, which stays within closed-form fitting on activations I actually read.

I'm weighing whether the 768 tanh pooler units can stand in for the quadratic kernel, since the full C² feature space would be far too large. With about 1,500 training points, a Nyström-style set of centers might capture most of the benefit. I want to check that in numpy on read activations first, then confirm with a real forward read.

I'm sketching the pooler's 768 tanh units as random features over the hidden state, with a 3072-wide FFN in the pool slot feeding them. I'm checking random Gaussian projections against training-point centers, since the center version may be slightly optimistic even though it uses no labels.

I'm considering whether the pooler's 768 tanh units, fed by random or center-based projections of the layer-9 state, can recover most of the quadratic kernel's benefit. I'll test that on read activations in numpy first, then confirm with a real forward read.

I'm checking whether the LayerNorm in the final layer, which normalizes each vector across its dimensions, changes how the pooler's centering and scaling can be absorbed into its weights. If the FFN output is switched off, the CLS state is just the LayerNorm of the attention output plus bias, so I'm working out what that means for the features I can compute on read activations.

I'm planning to read the last layer's CLS and mean states for all 1,496 practice ids in two passes, since one read caps at 1,024. Then I'll fit the pooler's random features and the classifier in numpy and check them by CV before writing the weights and verifying with a predict pass. I still need to decide whether the reserved channel can carry the M2 information once layers 0 through 8 are kept.

I'm now checking whether the carry heads at layers 2 and 6 still hold the M2 and M6 ridge scores. Those were fit on all the data, which would leak into CV, so I'm considering refitting them on the training fold only, with the M2 projection rebuilt from scratch.

I'm now working on the carry heads: rather than reuse the joint ridge fit on all data, I'd refit the L2 carry weights myself on the M2 mean so the g(M2) term can be scored honestly under CV.

I'm checking whether the carry head can write the reserved dims with g(M2), scaled, so the pooler sees both the M9 kernel features and a linear M2 term. The catch is that g was fit on all the training data, which would leak into CV, so I'd need to refit it on each training fold to score it honestly. Since M2 adds about 0.2 points on top of the M9 quadratic alone, it seems worth keeping.

I'm weighing whether to push further by dropping layers or shrinking the feed-forward width. Dropping to m=1536 cost about 0.75 points earlier, and m=2560 would likely lose something too, so I'm keeping k=10 with m=3072 as the final design.

I'm weighing whether the CLS state at layer 9 adds anything beyond the mean-pooled state, so I'm planning to test combining it with the mean before settling on k=10 or k=9.

I'm checking whether a kernel on the layer-8 state, or a sum of kernels over several layers, could let the network drop to k=9, but the reserved channel only carries linear scores, so earlier nonlinear features can't pass through it.

I'm wondering whether the reserved channel could carry more than its 8 dimensions, but widening it would damage the network further, since the earlier edit already zeroed those dims and the base reads reflect that loss. I'll check the C9 contribution and a few other layer combinations quickly, including quadratic features on M8 paired with C8.



Depth 9 reaches about 97.95% CV, above the 97.8% bar, while depth 8 tops out near 97.5%, which is too risky. I'll check whether CLS vectors or multi-layer kernels can lift depth 8 before committing.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom cvlib import folds\nd=np.load('base_reads.npz'); y=d['y']; R=np.load('reserved.npy')\nkeep=np.setdiff1d(np.arange(768),R)\nM={j:d[f'mean_hidden_L{j}'][:,keep] for j in range(13)}\nC={j:d[f'cls_hidden_L{j}'][:,keep] for j in range(13)}\ndef z(X): return (X-X.mean(0))/(X.std(0)+1e-6)\ndef kcv(K, y, lams, seeds=(0,1,2)):\n    Y=np.eye(8)[y]; out={}\n    for lam in lams:\n        a=[]\n        for s in seeds:\n            f=folds(y,5,s); S=np.zeros((len(y),8))\n            for i in range(5):\n                tr=f!=i; te=f==i\n                A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0))\n                S[te]=K[np.ix_(te,tr)]@A\n            a.append((S.argmax(1)==y).mean())\n        out[lam]=round(100*np.mean(a),2)\n    return out\ndef quad(X,c=0.5):\n    Z=z(X); Z/=np.linalg.norm(Z,axis=1,keepdims=True); Cc=Z@Z.T; return Cc+c*Cc**2\ndef lin(X): Z=z(X)/np.sqrt(X.shape[1]); return Z@Z.T\nL=(1e-2,3e-2)\nprint('M8 quad(M8+C8)+linM2', kcv(quad(np.hstack([M[8],C[8]]))+lin(M[2]),y,L))\nprint('M8 quad(M8,c=1)+linM2', kcv(quad(M[8],1.0)+lin(M[2]),y,L))\nprint('M8 quad(M8)+quad(M5)+linM2', kcv(quad(M[8])+quad(M[5])+lin(M[2]),y,L))\nprint('M8 quad(cat M8 M6)+linM2', kcv(quad(np.hstack([M[8],M[6]]))+lin(M[2]),y,L))\nprint('M8 quad(M8)+linM2+linM5', kcv(quad(M[8])+lin(M[2])+lin(M[5]),y,L))\nprint('M9 quad(M9+C9)+linM2', kcv(quad(np.hstack([M[9],C[9]]))+lin(M[2]),y,L))\nprint('M9 quad(M9)+linM2+linM6', kcv(quad(M[9])+lin(M[2])+lin(M[6]),y,L))\nprint('M9 quad(M9,c=1)+linM2', kcv(quad(M[9],1.0)+lin(M[2]),y,L))"
}
```

