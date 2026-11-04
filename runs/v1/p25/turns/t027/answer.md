## reasoning summary

The implementation agrees with the offline model to within about 1%, which fits the expected error from contamination and from rounding the logits to four decimals in the JSON. The ratio range of 0.24 to 1.07 corresponds to a σ between 0.94 and 4.1, which is acceptable. The 1% maximum residual also looks explained by that rounding plus contamination.

Next I want to lift the baseline's accuracy by tapping layer 6, so I'm planning to adjust that layer's query weights and then re-read the layer 11 states for all 1496 samples.

I'm computing the direction d by minimizing its quadratic form under the constraint that it projects to 1 on the mean CLS state, then checking how d projects onto the token states and the CLS states, to see whether the layer 6 edit stays small on the tokens.

I'm weighing whether tapping several layers at once would beat a single tap at layer 6, since the CLS states would drift after the earlier edits and throw off the uniform-attention trick.

I'm planning a first experiment that taps layer 6 and then reads the CLS states at layers 7 through 11, plus the mean-pooled layer 11 states, to see how far the token states shift. I also need to account for the fact that the CLS states at deeper layers will have moved after the layer 6 edit, which could undercut the uniform-attention trick for later taps.

I'm going to compute the layer 6 direction and apply the tap, then read the token states at layers 4, 7, 8, 9, and 10 in one pass over the 64 texts, which keeps the memory cost to roughly 53MB.



The baseline is confirmed at roughly 96.7% expected accuracy, since argmax matches on all 300 items. Next I'll test a layer-6 [CLS] tap by rewriting W_q, starting with diagnostics on the direction d.



## tool_use: run_python

```json
{
 "code": "F=np.load('feats.npz'); T=np.load('tok128.npz'); sid=T['sid']\n# non-CLS token states at L6: first token of each sentence is CLS\nfirst=np.r_[True, sid[1:]!=sid[:-1]]\nH6=T['T6'][~first]; Hc6=T['T6'][first]\nC6=F['C6']; Cbar=C6.mean(0); D=C6-Cbar\nprint('cls from tok128 vs feats check', np.abs(Hc6.mean(0)-Cbar).max().round(3))\nfor lam_r in [0.1,1.0]:\n  for wcls in [1.0,5.0]:\n    S=H6.T@H6/len(H6) + wcls*D.T@D/len(D) + lam_r*np.eye(768)*0.01\n    d=np.linalg.solve(S,Cbar); d/= d@Cbar\n    dt=H6@d; dc=C6@d\n    print(lam_r,wcls,'|d|',np.linalg.norm(d).round(3),'tok d.h mean %.4f std %.4f absmax %.3f'%(dt.mean(),dt.std(),np.abs(dt).max()),' cls d.h mean %.3f std %.4f'%(dc.mean(),dc.std()))"
}
```

