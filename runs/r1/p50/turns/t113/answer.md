## reasoning summary

The decomposition splits the total into layers 0–1 at about 48%, layers 2–4 at about 37%, and the remaining 15% from layer 5 plus interactions. Layers 0 and 1 each account for roughly a quarter, and 2, 3, and 4 about 12% apiece. Improving layers 3 and 4 by the 10–22% seen locally might cut the total by around 3–4 points.

So the early layers, L0 and L1, dominate the error, so that's where improvement matters most. For those I'm considering greedy selection, which only gave L1 about 2%, and more data, since L0's input is base-exact and I already have 96 sequences from it.

I'm weighing whether dropping a middle layer, such as L2 or L3, would hurt less than dropping L4, since only L4 has been tested so far. Each such test would cost a pass plus a head refit.

I'm considering whether dropping a middle layer like L2 or L3 would cost less than dropping L4, but only L4 has been tested so far. A drop costs roughly .955–.97 agreement, which is worse than the m1536 pruning at 68.1M, so a 5-layer model at m2048 would stack both losses and seems unlikely to win.

Sizing up whether dropping a layer beats shrinking everything, I'm checking how much noise each layer adds at m=1280 versus m=1536. The ratios run about 1.31 to 1.44 across layers, with L0 the worst.

I'm working out how the flip count shifts with model width. Each step up or down in width scales the per-layer error, and the flips follow roughly linearly with that error. Going from 1536 to 2048 only trims flips by about a third for roughly 7% more parameters, so the gain looks modest.

I'm weighing how the risk of failing the test changes with the number of flips. Assuming test flips scale with practice flips, a width of 1536 gives a failure chance of roughly 15–22%, and I'm checking how that shifts at 1280, where the expected flip count rises to about 19.

Comparing the per-flip risk across widths, m2048 looks safest at about 13.5%, while m1024 is worst at 36%. Scoring by expected loss times the remaining parameter fraction, though, the smaller widths come out roughly even with or slightly ahead of m1536, which makes me doubt the flat per-flip loss assumption.

I'm working through how the flip noise behaves: two models differing by N random flips disagree in accuracy with a standard deviation of roughly sqrt(N(p1+p2))/400. With N=16 and p1+p2 near .85, that spread is about 0.9 points, so any edited model carries a real chance of dropping more than a point below base.

I'm checking whether the risk stays near 15–20% even at m1536, since reaching under 5% would need much tighter fidelity and only save about 5% of parameters. I'm also wondering whether my test-to-practice flip ratio is overstated, because the margin density near zero may differ on the test set.

I'm weighing the risk of the F1 bar as well, since macro F1 on the small governance class could swing more than accuracy does. A drop of about 0.02 in F1 might correspond to only a handful of bad flips involving that class, and in practice roughly a third of the flips touch governance.

I'm weighing whether the risk can be lowered without giving up size, since even a modest model edit leaves a real chance of falling outside the bar. One idea is to refit the classifier head in closed form against the gold labels, so the edited model is nudged toward correct answers on borderline cases rather than only matching the base predictions.

I'm weighing whether a closed-form refit of the head against gold labels would be allowed and useful, but it could just mimic the base on training data and I have no way to validate it on test. Tuning the head's bias for macro-F1 has the same validation problem. So I'm leaning toward reducing variance by keeping the edited model's predictions closer to the base near decision boundaries.

Weighing whether to spend the remaining passes on calibrating m1280 and m1792 or on greedy selection for L0 and L4, since L0 and L1 carry about half the noise.

I'm checking whether layer 0's FFN output could be captured by a linear map on its input, and whether the token id alone explains most of that input's variance. Earlier linear fits didn't help, so I'm wondering if the dropped units' effect on L0 can be folded into the attention output projection instead.

I'm wondering whether the dropped units are simply independent features, which would mean L0 and L1 can't be compensated well. I'll test greedy selection on L0 and L2, plus weighted variants on L3, and if greedy works as well on L0 as it did on L4, I'll use it throughout.



Noise is split roughly 48% in L0–1, 37% in L2–4, and about 15% in L5 and interactions, so L0 and L1 each carry about a quarter. That's why the L3/L4 gains went unseen; I'll test greedy selection on L0, L2 and L3, including a post-LN-scaled variant.



## tool_use: run_python

```json
{
 "code": "exec(open('seq.py').read())\nimport time\nrng=np.random.default_rng(5)\ndef greedy_select(Hc_w, W2e, m, delta=0.05):\n    S=Hc_w.T@Hc_w/len(Hc_w); S+=delta*np.diag(np.diag(S))\n    W2S=W2e@S; chosen=[]; alive=np.ones(S.shape[0],bool)\n    for it in range(m):\n        d=np.diag(S); g=(W2S**2).sum(0)/np.maximum(d,1e-12); g[~alive]=-1\n        j=int(np.argmax(g)); chosen.append(j); alive[j]=False\n        s=S[:,j].copy(); sj=s[j]\n        S-=np.outer(s,s)/sj; W2S-=np.outer(W2S[:,j],s)/sj\n    return np.sort(chosen)\nfor L in [0,2,3]:\n    A,S_=all_data(L); A=A.astype(np.float64)\n    useq=np.unique(S_); te_seq=rng.choice(useq,len(useq)//4,replace=False); te=np.isin(S_,te_seq); tr=~te\n    pre=f\"roberta.encoder.layer.{L}.\"\n    W1=base(pre+\"intermediate.dense.weight\").astype(np.float64); b1=base(pre+\"intermediate.dense.bias\")\n    W2=base(pre+\"output.dense.weight\").astype(np.float64); b2=base(pre+\"output.dense.bias\"); g2=base(pre+\"output.LayerNorm.weight\").astype(np.float64)\n    H=gelu(A@W1.T+b1); U=A+H@W2.T+b2; sig=U.std(1,keepdims=True)\n    m=1536; Htr=H[tr]; Hc=Htr-Htr.mean(0); w=1/sig[tr]**2\n    kv=np.sort(np.argsort(-(Htr.var(0)*(W2**2).sum(0)))[:m])\n    kg=greedy_select(Hc,W2,m)\n    kgw=greedy_select(Hc*np.sqrt(w/w.mean()), g2[:,None]*W2, m)\n    for name,keep in [('var',kv),('greedy',kg),('greedy_w',kgw)]:\n        drop=np.setdiff1d(np.arange(3072),keep)\n        T=H[:,drop]@W2[:,drop].T; X=H[:,keep]; mu=X[tr].mean(0); Xc=X[tr]-mu; G=Xc.T@Xc\n        B=np.linalg.solve(G+0.1*np.trace(G)/m*np.eye(m), Xc.T@(T[tr]-T[tr].mean(0)))\n        E=T[te]-((X[te]-mu)@B+T[tr].mean(0)); en=(((E-E.mean(1,keepdims=True))/sig[te]*g2)**2).sum(1).mean()\n        print(L,name,'test %.3f'%en, flush=True)\n"
}
```

