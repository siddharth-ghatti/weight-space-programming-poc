## reasoning summary

Correcting at the end lowers the mean |dlogit| a little but makes agreement worse, so it doesn't help. Since refitting the head didn't fix it either, the noise from layers 0–4 looks like it's baked into the information itself. The only real lever is a better approximation in those early layers. Next I want to see how the noise shrinks as I add units, comparing the 1024-unit setup against the full 1536.

I'm weighing whether more data would help the early-layer fits. Doubling the data might trim the error by roughly 10-15%, but each extra pass only adds about 3-4k tokens because the longest ids are already used, so the return looks poor. I'm also checking whether the token-level limit of 32 ids per pass holds for both token and attention outputs, and whether the mean-pooled outputs could help, though they seem too coarse for nonlinear fitting.

I'm checking whether the objective should weight tokens differently, since the final CLS depends on them through attention. The fits are data-limited, so the error may simply be what it is. I'm also considering whether a different structure, like dropping a layer and widening the rest, would reduce the noise, though dropping layer 4 alone already hurt agreement a lot, so that seems too coarse.

I'm weighing whether to size up the m-choice on a more concrete basis. The m1536 setup has 19 flips in-sample, about 1.9%, which worries me given the test risk. I'm considering a variance model with an assumed test distribution, possibly by scaling the practice base logits so the base test accuracy lines up with 77.5%, though I don't know the actual test margin distribution, so that's only a rough proxy.

I'm estimating how the flip risk depends on margin: flips cluster at small base margins, so the practice rates by bin give a rough per-flip likelihood. I'd expect test to have more mass at low margins than practice, since the base model's errors concentrate there, though I'm only guessing at those shares.

I'm estimating the flip count for the m1536 setup: about 8 to 14 flips on test, depending on whether I weight the margin bins by practice or by my guessed test mix. Near margin zero, the base model is right maybe half the time and the alternative about a third, so each flip carries a small expected loss with real variance.

I'm checking the pass-or-fail threshold now: the base model is at 310 of 400, so falling to 306 or below means a net loss of five or more, and I'm estimating the chance of that from the flip counts and per-flip variance.

I'm working out the pass threshold: the base is at 310 of 400, so a net loss of five items would drop it to 305 (76.25%), which fails, while a net loss of four still lands at 306 (76.5%) and passes. The macro-F1 condition is a separate concern, since small classes like governance could swing the score noticeably from a single flipped item.

I'm weighing whether to improve the fidelity of the m1536 fit instead of accepting its flip risk, since fewer flips would lower the chance of failing the accuracy threshold.

I'm thinking about whether a smarter selection of kept units could help, since the current variance-based ranking ignores redundancy between units. One option is a subset selection that accounts for how well the remaining units can compensate for the dropped ones, though this is still an open idea.

I'm still on the selection question: greedy forward selection is the standard fix for redundancy, though the earlier OBS-style attempt didn't beat variance ranking on holdout, which I suspect is overfitting given how little data there is. A different angle would be choosing units by how much they matter downstream rather than by local reconstruction error, but I don't know their sensitivity yet. Another idea is to get more training samples for the layer fits by generating plausible FFN inputs, though I haven't worked out how to do that.

I'm wondering whether synthetic activations, say Gaussian samples matched to the empirical covariance, could help the layer fits. The rules seem to allow data computed from practice activations, but the regression error depends on the true activation distribution, and token activations are heavily structured, so a Gaussian stand-in would likely mismatch it.

I'm considering whether to add noise or mixup to the training activations, so the ridge fit doesn't latch onto the exact samples and generalizes better to held-out tokens. The uncertainty is whether mixed or noised inputs stay close enough to the real activation distribution to help, so I'd test it on the holdout first.

I'm weighing whether a bigger m would be worth it, since the per-flip risk at m1536 looks too high. With the gap to held-out error, better regression alone won't buy much, so the number of kept units is the main lever. The attention side can't be pruned, so the choice comes down to m.

I'm weighing whether a smarter choice of kept units or more data for the layer fits could cut the noise, but the holdout gap suggests better regression alone won't buy much. I'm comparing the m1024, m1280, and m1536 options against the flip risk, and I'm still thinking about the effect of the lam setting on holdout error.

I'm estimating the flip counts for the smaller widths from the noise scaling: m2048 would drop to roughly 13 to 14 practice flips, and m1280 and m1024 would land somewhere in the low-20s to mid-20s, though the offline m1024 estimate of about 36 flips suggests that scaling underestimates the loss at smaller widths.

Now I'm estimating the flip counts at each width: about 19 at m1536, 36 at m1024, and somewhere around 12 to 16 at m2048. Scaling the practice rates to test, I get roughly 11 flips at m1536, 22 at m1024, and 8 at m2048, and I'm about to work out the failure odds for m2048 from its mean and spread per flip.

I'm working through the expected flip losses for each width: about 1.7 at m1536, with a 3.1 spread, giving roughly an 18% chance of failing. The smaller widths look worse, with m1024 landing near 39%. I'm now questioning whether the per-flip variance of 0.85 is too pessimistic, since the outcomes near the boundary may be less spread out than I assumed.

I'm checking the per-flip assumptions: among the practice flips the base is right on most of them, and I'm guessing the test-set split is about even between the two classes. The failure odds stay high for every width, and the gap between m1536 and m2048 is only a few points under this rough model. Scored linearly, the options come out nearly flat.

I'm wondering whether one of the early layers is responsible for most of the noise, since its errors would carry through all five layers. If so, the uniform width might be the wrong allocation, though I can't give a single layer more units without changing the setup. I'll check per-layer contributions at m1536 before deciding.

I'm wondering whether a layer's spare units could do something useful, but the uniform-width constraint means every kept layer gets the same m, so there's no saving to reallocate. Layer 5's FFN can only act on the CLS output after its attention, and since that output isn't correctable, the extra capacity there wouldn't help.

Now I'm checking whether the early-layer noise is concentrated on the tokens that the final CLS attends to in layer 5, since those would be the ones worth weighting in the compensation.

I'm checking whether the compensation error is larger at the CLS token than at regular tokens, since CLS has that huge constant dimension and is only about one token per sequence in the fitting data. If it is, I'd weight CLS samples more heavily in the fit.

I'm wondering whether the CLS token has larger compensation error than regular tokens, since it carries that huge constant dimension and appears only once per sequence in the fitting data. The full set of CLS inputs per layer could come from the probe passes, but a single pass can't hold both the token ids and the 1000 CLS ids, so that would take separate passes. First I'll check the CLS-versus-other error gap offline on the existing holdout data.



End-to-end correction through layer 5 makes agreement worse, so the upstream noise can't be fixed linearly at the end. I'm checking whether compensation is poor specifically at `<s>` positions, since those feed the classifier through the residual stream.



## tool_use: run_python

```json
{
 "code": "exec(open('seq.py').read())\nrng=np.random.default_rng(1)\niscls0=np.load('ffn_in_L0_iscls.npy').astype(bool)\nfor L in range(5):\n    A,S=all_data(L); A=A.astype(np.float64)\n    # cls flag: first token of each sequence (S changes)\n    iscls=np.r_[True, S[1:]!=S[:-1]]\n    useq=np.unique(S); te_seq=rng.choice(useq,len(useq)//4,replace=False); te=np.isin(S,te_seq); tr=~te\n    pre=f\"roberta.encoder.layer.{L}.\"\n    W1=base(pre+\"intermediate.dense.weight\").astype(np.float64); b1=base(pre+\"intermediate.dense.bias\")\n    W2=base(pre+\"output.dense.weight\").astype(np.float64); b2=base(pre+\"output.dense.bias\"); g2=base(pre+\"output.LayerNorm.weight\")\n    H=gelu(A@W1.T+b1); F=H@W2.T; U=A+F+b2; sig=U.std(1,keepdims=True)\n    m=1536; sc=H[tr].var(0)*(W2**2).sum(0); order=np.argsort(-sc); keep=np.sort(order[:m]); drop=order[m:]\n    T=H[:,drop]@W2[:,drop].T; X=H[:,keep]; mu=X[tr].mean(0); Xc=X[tr]-mu; G=Xc.T@Xc\n    B=np.linalg.solve(G+0.1*np.trace(G)/m*np.eye(m), Xc.T@(T[tr]-T[tr].mean(0)))\n    E=T-((X-mu)@B+T[tr].mean(0)); Eo=(E-E.mean(1,keepdims=True))/sig*g2\n    en=(Eo**2).sum(1)\n    print(L, 'n_cls', iscls.sum(), 'post-LN err energy: cls test %.3f  tok test %.3f | cls train %.3f tok train %.3f'%(en[te&iscls].mean(), en[te&~iscls].mean(), en[tr&iscls].mean(), en[tr&~iscls].mean()),\n          ' sigma cls %.1f tok %.1f'%(sig[iscls].mean(), sig[~iscls].mean()))\n"
}
```

