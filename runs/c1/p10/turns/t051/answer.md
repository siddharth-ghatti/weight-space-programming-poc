## reasoning summary

GELU features reach about 98.17 at best, roughly matching RBF, so there's no real gain over KRR, though it edges out the 768-center pooler SoR by 0.1. It's also sensitive to hyperparameters, since g=3 hurts. A notable difference is that the FFN and pooler heads could be stacked, with the FFN's GELU units feeding scores into reserved dimensions.

Weighing a few options for reducing the model: at k=10, keeping all 3072 units gives about 95.3M parameters with accuracy roughly at baseline, while trimming to 2560 units drops it to about 87.4M at roughly half a point lower. Dropping the early layers looks like it would wreck the representation, so I'm setting that aside.

I'm weighing whether the Gaussian proxy could be improved with the activation data I already have, since reading FFN statistics directly through an edited model won't work: changing one layer's output would corrupt the inputs to everything after it.

Weighing whether a layer-by-layer pruning could beat the uniform cut: the risk is that the cost of removing FFN units might differ between layers, though every layer uses the same width right now. Fewer layers with a richer head is another option, since the 97.9 and 98.06 results are close enough to be worth comparing against the 95.3M baseline.

Checking whether L9's jump over L8 holds up, since a gap of about 13 samples in LOO could just be noise. Combining M9 with M0 in a kernel looks promising, with a CV of 98.5 against 98.2 for M9 alone.

Checking whether the pooling could be moved to a different layer, I'm considering if a layer-0 head could gather the M0 signal at the CLS position. Any head there would average values across all tokens and disturb the representations, so it looks unworkable unless the write-out stays confined to reserved dimensions, which the layer-1 zeroing doesn't guarantee. The carry channel would have to hold about eight numbers, the ridge scores from M0 or M1.

I'm checking whether combining the M0 scores with K9 actually helps under LOO and CV, and whether the layer-2 carry head could supply M2 scores for the same purpose. The concern is that the carry channel's current ridge scores from M2 and M6 were fitted jointly with the final CLS, so they may not behave well on their own. Next I need to work out how the additive model would be realized, since the pooling layer's LN zeroes out the reserved dimensions.

I'm working out how to keep the reserved dimensions passing through as a carry, so the final output is the carried scores plus the kernel scores. That would make the additive combination of the linear M2 term and the kernel exact. I want to check empirically whether K9 with lin(M0), lin(M1), or lin(M2) actually helps under CV, fitting jointly as a kernel ridge with an added linear term.

Now I'm checking whether the additive K9 + lin(M0) gain of about 98.5 against 98.2 for M9 alone holds under three-seed CV. If lin(M2) does as well, the layer-2 carry head could be refit to write ridge scores for M2, since its input is the layer-1 output.

I'm checking how the reserved dimensions at the layer-8 output behave, since each LayerNorm rescales them per token, so the carried values may differ across tokens. The plan is to read those dimensions directly and see how they relate to the M2 and M6 ridge scores.

I'm checking whether adding M0 as a linear term to K9 gives a real gain over M9 alone, testing the same for M1 and M2 to compare. The M2 and M6 carry scores already tested as linear terms added nothing, so the open question is whether M0 can be made available at all, since the carry heads at layer 2 and above can't reach the embedding mean.

Checking whether M0 is truly out of reach for the carry heads, since layer-1 outputs do hold embedding information in transformed form. A linear ridge on M2 reaches about 96%, close to M0's 96.2, but its contribution alongside K9 might differ. I'm about to test additive models that combine K9 with a linear term from M0, M1, or M2, using a linear kernel scaled by a tuned constant, and I'll also look at the reserved dimensions at L9 for the existing carry.

I'm working out how a linear term carried through the reserved dimensions could be added to the kernel scores, since a summed kernel gives a fit that splits cleanly into a kernel part and a linear part.



GELU-at-all-centers reaches about 98.17, roughly level with RBF KRR and only 0.1 above the pooler head, so a better head won't recover the 0.5 lost to pruning. Next I'll test whether a linear term from shallow layers adds margin, comparing M0, M1, M2 and the existing channel with 3-seed CV.



## tool_use: run_python

```json
{
 "code": "exec(open('krr.py').read()); exec(open('lib.py').read())\nd=np.load('base_feats.npz'); y=d['y']\nF=lnr(5*d['mean_hidden_L9']+d['cls_hidden_L9'])\ndef cv_sum(F, Lin, y, c, gamma=0.4, lam=0.003, seeds=(0,1,2)):\n    accs=[]\n    for s in seeds:\n        P=np.zeros((len(y),8))\n        for te in folds(len(y),5,s,y):\n            tr=np.setdiff1d(np.arange(len(y)),te)\n            Ztr,Zte=zs(F[tr],F[te])\n            Ktr=rbf(Ztr,Ztr,gamma); Kte=rbf(Zte,Ztr,gamma)\n            if Lin is not None and c>0:\n                Ltr,Lte=zs(Lin[tr],Lin[te]); dl=Lin.shape[1]\n                Ktr=Ktr+c*Ltr@Ltr.T/dl; Kte=Kte+c*Lte@Ltr.T/dl\n            T=np.eye(8)[y[tr]]; T=T-T.mean(0)\n            al=np.linalg.solve(Ktr+lam*np.eye(len(tr)),T); P[te]=Kte@al\n        accs.append((P.argmax(1)==y).mean())\n    return np.mean(accs)\nprint('K9', cv_sum(F,None,y,0))\nfor nm in ['mean_hidden_L0','mean_hidden_L1','mean_hidden_L2','mean_hidden_L4']:\n    Lin=d[nm][:,KEEP]\n    print(nm, [round(cv_sum(F,Lin,y,c),4) for c in [0.03,0.1,0.3]])\ncarry=d['mean_hidden_L9'][:,RES]\nprint('carry9', [round(cv_sum(F,carry,y,c),4) for c in [0.03,0.1,0.3]])\n"
}
```

