## reasoning summary

The partial result is in: C7' scores better than the baseline but still falls short of M6's 96.5%. By layer 11 the CLS signal converges back, and combining C11' with M11' gives no gain over the base design. Token perturbation shifts the mean features by about 4-5%, which I'd call moderate, and M11' accuracy holds steady.

I'm weighing three ways to strengthen the signal: tapping at layer 10, shielding the CLS channel from dilution in layers 7-10, or tapping at every layer from 6 to 10, which I suspect would just reproduce what M11 already carries. I'm also wondering whether a much stronger tap at layer 6 could keep the M6 information intact through to the end.

I'm considering whether a stronger tap at layer 6 could help, but attention updates change CLS and tokens together, so scaling the value path isn't a clean option. Amplifying CLS through a gated FFN at layer 6 would only boost the state within that layer, since LayerNorm rescales it afterward, and each later layer's attention and FFN output still dilutes the injected information.

I'm weighing whether attention-driven dilution can be avoided, since shielding CLS from later layers' additions looks impractical. I'm also considering the SEP token as a possible sink, since tokens attend to it heavily in the middle layers and its values may be near-constant. Another angle is to tap a second, less diluted path instead of boosting CLS directly.

I'm considering whether a tap at layers 9 or 10 could work, since it would read only the current layer's tokens. Another option is to have the CLS pool at every layer from the tap onward, though earlier contributions would likely be swamped by later ones. I'm also looking at protecting the CLS channel by having it attend only to itself in layers 7-10, which would make its attention output a linear function of its own state rather than something that dilutes the injected information.

I'm working out how to keep CLS's state from being diluted in layers 7-10: if CLS attends only to itself there, its attention output becomes a linear function of its own state, so the injected M6 information survives, only nonlinearly transformed by the FFN.

Working out the mechanism now: I'd tune the query and key weights in each head of layers 7 through 10 so that the CLS token attends mostly to itself, which would keep its own state from being diluted by the other tokens' contributions.

I'm checking the attention scores for CLS against tokens: the self-term from the β-scaled query should dominate, but tokens' keys also pick up a β·(u·k_j) term that I need to keep small, so I'm looking for a head direction u where token keys are nearly orthogonal to it.

I'm trying to make CLS attend mostly to itself in layers 7-10 by scaling the query and key weights with a shared direction. The catch is that scaling the key direction also changes how tokens attend to CLS, so I need to check that side effect too.

I'm picking u to point against the typical token query, so tokens' queries are pushed away from CLS's key and they stop attending to it. The open question is whether the β·(u·k_j) term that CLS picks up from the tokens' keys stays small next to the self-attention term.

I'm checking whether shifting the query and key weights by a shared direction u, with a large α on the key side, would make CLS attend to itself while tokens stop attending to it. The token-side perturbation looks small, about 4% on queries and a similar amount on keys, so it seems acceptable.

I'm checking whether keeping the CLS state intact through layers 7–10 pays off: I'd compare a concatenation of C7' with M11' against the baseline, using C8' and C9' as well, to see if the preserved M6 information actually adds accuracy.

I'm running the same cross-validation for the concatenation of C10' with M11', to see whether keeping the layer-10 CLS state adds anything beyond M11' alone.



The layer-6 tap works locally, lifting [CLS] at layer 7 to 94.9% CV, but the gain washes out by layer 10, leaving the final concat at about 96.7%. Next I'll check how good [C_l′, M11′] is at earlier layers, since that caps any "protect [CLS]" fix.



## tool_use: run_python

```json
{
 "code": "F=np.load('feats.npz'); y=F['y']; folds=np.load('folds.npy'); A=dict(np.load('featsA.npz'))\ndef stdz(X,tr): mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; return (X-mu)/sd\ndef cv(featfn, lam):\n    accs=[]\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        Z=featfn(tr); mu=Z[tr].mean(0); Zc=Z-mu\n        Y=np.eye(8)[y[tr]]-1/8\n        K=Zc@Zc[tr].T; a=np.linalg.solve(K[tr]+lam*np.eye(tr.sum()),Y); pr=K[te]@a\n        accs.append((np.argmax(pr,1)==y[te]).mean())\n    return round(np.mean(accs),4)\nfor k in ['C7','C8','C9','C10']:\n    fn=lambda tr: np.concatenate([stdz(A[k],tr),stdz(A['M11'],tr)],1)\n    print(k,\"'+M11'\",[cv(fn,l) for l in [100,200,300,500]])\nfn=lambda tr: np.concatenate([stdz(F['C7'],tr),stdz(F['M11'],tr)],1); print('base C7+M11',[cv(fn,l) for l in [100,200,300,500]])\nfn=lambda tr: np.concatenate([stdz(F['M6'],tr),stdz(A['M11'],tr)],1); print(\"M6+M11'\",[cv(fn,l) for l in [100,200,300,500]])"
}
```

