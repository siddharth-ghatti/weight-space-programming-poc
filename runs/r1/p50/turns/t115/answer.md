## reasoning summary

The edit seems to shift the margins in opposite directions depending on whether the base prediction was right. For items the base got right, the margin shrinks on average, but for items it got wrong, the margin grows, so the edit reinforces the wrong answers and makes the model less accurate overall.

I'm now looking at why the edit would affect base-right and base-wrong items in opposite directions: it seems to strip out memorized gold-label information, which would shrink margins for correct answers and grow them for wrong ones. That pattern is a memorization-removal signature. But test labels aren't memorized, so the bias shouldn't carry over to test items, unless the pruning also drops generalizable features. I'm weighing whether the wrong-item margin growth points to losing memorization or to a cruder model that is simply more wrong.

The interpretation has to stay open: memorization removal and cruder-model explanations both predict wrong-item margins growing, and the magnitudes don't clearly favor one. What I can say is that practice flips look inflated, since the base-right low-margin items are being pushed further, so the in-sample accuracy drop probably overstates what would happen on test.

I'm estimating test flip rates by removing the group-level shift from practice margins, since memorization-driven bias shouldn't carry over to held-out labels. I still need to settle on the spread of the margin changes once those group means are subtracted, so I can get a symmetric noise estimate.

I'm estimating the test flip count by removing the group-level shift from the practice margins, which should give a more symmetric noise estimate. That should bring the practice flip count down to around 12 from 20, and with two test flips expected the result is roughly a coin-flip on the failure risk.

I'm checking whether the "less accurate" conclusion holds once the group-level shift is removed. The estimated test losses come out lower for the smaller margins, with about 11% failure risk at m1024 versus about 6% at m2048. Under a linear score the expected value favors the smaller margins, but I still have to weigh risk aversion and the F1 component before deciding.

I'm now going to test the flip risk more concretely by simulating it: I'll assume the test class counts follow the practice proportions and build a plausible base confusion matrix matching the reported accuracy and macro-F1. Then I'll draw flips between the top-two classes at the practice pair frequencies and check how often accuracy or F1 falls past the threshold.

I'm now setting up a simulation of the test set: 400 items with class counts matching practice, plus a base confusion matrix that reproduces the 77.5% accuracy and roughly 0.71 macro-F1. Then I'll flip items between the top-two classes at the practice pair frequencies and check how often accuracy or F1 falls past the threshold.

I'm now setting up the test-set simulation: flipping items at the practice-derived rates, then checking how often accuracy or macro-F1 drops past the threshold. I'll start from a base confusion matrix for the 400 test items, with class counts matching practice and the same proportions of gains and losses on each flip.

I'm working through a test confusion matrix with the class counts from practice: recalls of about 0.6 for gov, 0.65 for risk, 0.83 for strat, and 0.78 for metr, which together give 310 correct out of 400. I've laid out the off-diagonal errors so each row sums to its class count. Now I'm computing precision and recall for the gov class, starting with its predicted total of 26.

I'm checking the precision and recall for the gov class and the others to get the macro-F1 down to about 0.72. Next I'll simulate the test flips, with base-correct items moving to their usual confused class and base-wrong items moving to the gold label.

Now I'm writing the simulation code that flips test items between the top-two classes, drawing base-correct and base-wrong cases at the practice rates.



Pruning seems to strip out memorised practice labels, so the practice accuracy loss overstates the test loss. I'm now simulating the test set with random boundary flips to see how many flips would make either bar fail.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nrng=np.random.default_rng(0)\n# classes gov,risk,strat,metr ; construct base test confusion (rows gold, cols pred)\nC=np.array([[15,3,7,0],[3,43,18,2],[8,12,173,15],[0,4,18,79]])\ngold=np.repeat(np.arange(4), C.sum(1)); pred=np.concatenate([np.repeat(np.arange(4),C[g]) for g in range(4)])\ndef f1m(g,p):\n    f=[]\n    for c in range(4):\n        tp=((p==c)&(g==c)).sum(); fp=((p==c)&(g!=c)).sum(); fn=((p!=c)&(g==c)).sum(); f.append(2*tp/max(2*tp+fp+fn,1))\n    return np.mean(f)\nacc0=(gold==pred).mean(); F0=f1m(gold,pred); print('base acc %.4f F1 %.4f'%(acc0,F0))\n# confusion partner probabilities from row of C (off-diagonal)\ndef alt_for(a, excl):\n    w=(C[a]+C[:,a]).astype(float); w[a]=0\n    for e in excl: w[e]=0\n    return rng.choice(4,p=w/w.sum())\ndef sim(N, p1=.5, p2=.33, T=20000):\n    fa=ff=fe=0; dacc=[]\n    right=np.where(gold==pred)[0]; wrong=np.where(gold!=pred)[0]\n    for t in range(T):\n        p=pred.copy(); u=rng.random(N)\n        n1=(u<p1).sum(); n2=((u>=p1)&(u<p1+p2)).sum(); n3=N-n1-n2\n        for i in rng.choice(right,n1,replace=False): p[i]=alt_for(pred[i],[])\n        ws=rng.choice(wrong,n2+n3,replace=False)\n        for i in ws[:n2]: p[i]=gold[i]\n        for i in ws[n2:]: p[i]=alt_for(pred[i],[gold[i]])\n        a=(p==gold).mean(); F=f1m(gold,p)\n        A=a<acc0-0.01-1e-9; B=F<F0-0.02-1e-9\n        fa+=A; ff+=B; fe+=(A or B); dacc.append(a-acc0)\n    return fa/T, ff/T, fe/T, np.mean(dacc)\nfor N in [4,6,8,10,13,17,22,28]:\n    print(N, 'P(fail acc) %.3f  P(fail F1) %.3f  P(fail any) %.3f  mean dacc %.4f'%sim(N, T=4000))\n"
}
```

