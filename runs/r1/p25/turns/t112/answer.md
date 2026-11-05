## reasoning summary

New data for the early layers barely lowered the error, so more data isn't the lever here, while width clearly helps: going to 1792 brings the error down to about 2.40. The layer-5 CLS error is much larger than the local error at that depth, so the attention there seems to amplify what it inherits. That leaves the question of what drives the error in layers 0 through 4.

I'm wondering whether a diagnostic pass could show which layer dominates the error in layers 0 through 4. A hybrid setup at m=3072 with zeroed columns would let me prune only L0–L2 while leaving the rest intact, though I'd still be stuck with uniform m, so the result would only point to where improvement effort should go.

I'm weighing a 2048 width against the smaller and larger options. Reusing the earlier 2048 run with the new data should leave it roughly unchanged, at about 15 flips and σ_low near 0.39. Running the rough estimates across 1536, 1792, 2048, and 2560 puts failure odds at about 31%, 24%, 21%, and 8%, though that model may be overcounting test flips.

Weighing the odds at each width now: 1536 looks like roughly a 14–31% chance of failing, and 1792 about 9–24%. I'm still trying to settle on the goal, which is the smallest model that keeps its accuracy.

I'm weighing whether a bigger hidden width could be swapped for an extra layer, but a 5-layer model at m=2560 comes out to about 71.3M parameters, which is worse than the 6-layer m=1792 option at 70.5M and drops a whole layer, so that's a poor trade. Reworking the attention in layer 5 to curb the error amplification is the last thing I'm considering.

I'm checking whether refitting the layer-5 attention output for the CLS token could cut its error, but a 768×768 refit on about 1000 samples would overfit, and the earlier head-correction test only gained a sliver. Going finer, to token-level attention in layers 3 and 4, would need far more token-level data than I can read per pass. I'm starting to think these methods have hit their limits, so I'm leaning toward settling on m=1792.

I'm weighing expected value across the widths: 1536 comes out ahead if a failure just scores zero, since its extra size savings outweigh the roughly 5–7 points of added failure risk compared with 1792. If a failure is costly, though, the safer 1792 or 2048 looks better.

I'm weighing 1664 as an intermediate width, which works out to about 69.3M parameters. Practice-set hidden states may differ from test-set ones, since the base model was trained on those items, so the flip counts I've been using might be skewed.

I'm checking whether the memorization in the practice set is hiding the real effect: on test items the drift should be more random, and pruning may even act as a regularizer that barely changes test accuracy.

I'm weighing which width to commit to, and the per-class flips for 1792 don't show any single class collapsing. Since the risk estimates overlap across 1536 and 1792, the choice comes down to how much I value the smaller size against the extra failure risk. I'll settle on 1792, since its practice agreement of about 98% holds up well against the earlier 2048 build.

Next I'm weighing whether to keep looking for a smarter way to preserve fidelity at a fixed width, since there are still a few passes and a couple hundred code runs left. The first idea, activation-aware unit selection with downstream propagation, doesn't seem worth pursuing, and bias shifting for the kept units has no closed-form fix. I'm now looking at folding the pruned units' contribution into the next layer's attention and feed-forward weights, starting from how a layer's output is built.

I'm considering whether the pruned units' leftover contribution can be folded into the residual path. The linear part of that contribution could be written as a matrix acting on the layer's input, and if that matrix were close to diagonal it would fit into the skip connection with no extra units. Earlier attempts with linear units didn't pay off, so the question is whether this route avoids that cost.

I'm checking whether the pruned units' leftover linear part could be folded into the layer norm gain and the next layer's first weight matrix, so the residual path absorbs it without extra units.

I'm checking whether the pruned units' linear leftover can be absorbed as a diagonal map, using the layer norm gain and the next layer's first weights so it costs no extra units. Since only the diagonal part of that map is free, I'm looking at whether including per-dimension regressors in the least-squares fit would capture it cheaply.

I'm testing whether the pruned units' leftover linear part can be captured diagonally, meaning a per-dimension term on each unit's own input activation, which the layer norm gain and the next layer's first weights could absorb at no extra cost. I'll check how much of the least-squares residual that per-dimension term explains at layers 0 through 2.

I'm going to test whether a per-dimension term on each output's own input activation can soak up the leftover linear part of the pruned units, using cross-validation to see if the gain is over 10% in relative error.



More data barely helped (upstream error 3.00 → 2.89), so width is the real lever. I'll test offline, with cross-validation, whether giving each output dimension its own input coordinate explains the remaining residual at L0–L2, since that costs no parameters.



## tool_use: run_python

```json
{
 "code": "\nimport sys; sys.path.insert(0,str(WORK))\nimport numpy as np\nfrom lsbuild import gelu, layer_data\ndef cv_diag(l, m, lam=100.0, wcls=20.0, weos=5.0, nf=4, lamd=1.0):\n    a,W1,b1,U,b2=layer_data(l,base)\n    if l<=2:\n        a2=np.load(f'true2_a_L{l}.npy').astype(np.float64)\n        tk=np.concatenate([np.load('tok_kind.npy'),np.load('tok2_kind.npy')])\n        grp=np.concatenate([np.load('tok_grp.npy'),1000+np.load('tok2_grp.npy')]); A=np.vstack([a,a2])\n    else:\n        tk=np.load('tok_kind.npy'); grp=np.load('tok_grp.npy'); A=a\n    G=gelu(A@W1.T+b1); Y=G@U.T+b2\n    w=np.where(tk==0,wcls,np.where(tk==1,weos,1.0))\n    ug=np.unique(grp); rng=np.random.RandomState(0); fo_map={g:i%nf for i,g in enumerate(rng.permutation(ug))}; fo=np.array([fo_map[g] for g in grp])\n    e0=[];e1=[]\n    for f in range(nf):\n        tr=fo!=f; te=(fo==f)&(tk==2)\n        wt=w[tr]; sw=np.sqrt(wt)[:,None]\n        mu=(wt[:,None]*G[tr]).sum(0)/wt.sum(); Gc=(G[tr]-mu)*sw\n        sc=(U**2).sum(0)*(Gc**2).sum(0); keep=np.sort(np.argsort(-sc)[:m])\n        ym=(wt[:,None]*Y[tr]).sum(0)/wt.sum(); Yc=(Y[tr]-ym)*sw\n        Xk=Gc[:,keep]; P=np.linalg.solve(Xk.T@Xk+lam*np.eye(m), Xk.T@Yc)   # m x 768\n        Rtr=Yc-Xk@P\n        # per-dim regressor a_j residualized on kept feats\n        am=(wt[:,None]*A[tr]).sum(0)/wt.sum(); Ac=(A[tr]-am)*sw\n        Q=np.linalg.solve(Xk.T@Xk+lam*np.eye(m), Xk.T@Ac); Ar=Ac-Xk@Q   # residualized a\n        md=(Ar*Rtr).sum(0)/((Ar**2).sum(0)+lamd)                         # per-dim coef\n        # prediction on test\n        Gte=G[te]-mu; Ate=A[te]-am\n        pred0=Gte[:,keep]@P+ym\n        pred1=pred0+(Ate-Gte[:,keep]@Q)*md\n        Yt=Y[te]; den=((Yt-Yt.mean(0))**2).sum()\n        e0.append(((pred0-Yt)**2).sum()/den); e1.append(((pred1-Yt)**2).sum()/den)\n    return np.mean(e0),np.mean(e1)\nfor l in [0,1,2]:\n    for m in [1536,1792]:\n        e0,e1=cv_diag(l,m); print(l,m,'plain %.4f  +diag %.4f  (%.1f%% better)'%(e0,e1,100*(1-e1/e0)))\n"
}
```

