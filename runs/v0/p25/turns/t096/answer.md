## reasoning summary

I'm dropping the idea of reusing the earlier route, since it gave no gain. Next I'm weighing whether the late-layer CLS-only approach could recover what the mean-pooled layer-0 features capture, namely lexical token identity that the layer-10 tokens only encode in mixed form. I'm considering a per-token nonlinear transform before pooling as a way to bring that identity back out.

I'm checking whether layer 10 could be repurposed to add per-token nonlinear features before pooling, since that might capture lexical identity that the mixed layer-10 tokens only partly encode. The open question is whether the base layer-10 contextualization would be lost, so I'm comparing how well cls10 plus mean10 performs against mean11 as a starting point. If it holds up near 0.98, the repurposing looks viable, and the per-token pooling could act like a set kernel that is more expressive than a kernel on the mean alone.

I'm weighing whether a per-token nonlinear step in layer 10 could recover lexical identity without losing the base contextualization, and I'm planning to compare cls10 plus mean10 against mean11 to see if the base holds up near 0.98. I'm also estimating whether the modest expected gain justifies the risk and the remaining passes.

I'm weighing whether layer 10 could be repurposed by keeping its attention sublayer intact and swapping in a new feed-forward block, but the original FFN already uses all 3072 neurons, so there's no room to add features without displacing existing ones. I'm also considering the kernel machine in layer 11, which currently uses 2039 centers, though alternative random-feature approaches don't look clearly better.

I'm considering whether a learned metric on the layer-11 features could beat simply reweighting the CLS and mean parts, since the kernel currently treats all directions equally. I'm also weighing an ensemble over several feature transforms at that layer, though the test set matches practice so the 0.98 CV is a fair guide.

I'm considering a learned metric instead: adding a closed-form LDA projection of the features to the kernel input, so the discriminative directions get emphasized. I'd fit the LDA inside each fold to keep the CV honest, then test whether it beats the plain kernel.



The kernel gets no gain from the layer-0 trick features, so that route is out. I'm next testing metric learning, stretching the LDA subspace inside the kernel, with nested CV so it's fitted only on training folds.



## tool_use: run_python

```json
{
 "code": "\nimport json, sys\nsys.path.insert(0,'.')\nfrom helpers import lda_fit\nfrom scipy.special import erf\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\nd=np.load('base_feats.npz'); y=d['y']; n=len(y)\ndef ln(x): x=x-x.mean(1,keepdims=True); return x/np.sqrt((x**2).mean(1,keepdims=True)+1e-12)\ndef stdz(X): return (X-X.mean(0))/(X.std(0)+1e-6)\nZ0=stdz(ln(d['cls11']+2*d['mean11']))\ndef run(gam, alpha=0.3, lam=0.01, seeds=3):\n    accs=[]\n    for seed in range(seeds):\n        f=np.random.default_rng(seed).permutation(n)%5; P=np.zeros((n,8))\n        for j in range(5):\n            tr=f!=j; te=~tr\n            Wt,b=lda_fit(Z0[tr],y[tr],alpha)\n            # orthonormal basis of discriminant directions\n            Q,_=np.linalg.qr(Wt.T)  # 768x8\n            Zp=Z0+gam*(Z0@Q)@Q.T\n            K=gelu(Zp@Zp.T/768)\n            T=np.eye(8)[y[tr]]; bb=T.mean(0)\n            A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()),T-bb); P[te]=K[np.ix_(te,tr)]@A+bb\n        accs.append((P.argmax(1)==y).mean())\n    return np.mean(accs)\nfor gam in [0,0.5,1,2]:\n    print(gam, round(run(gam),4))\n"
}
```

