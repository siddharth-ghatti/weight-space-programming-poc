## reasoning summary

The k-means centers reach 98.08% in 5-fold, close to the full KRR result of 98.2%, so a pooler-only head with 768 centers nearly matches it. Next I'll save notes and decide on the layer-drop exploration. For Experiment 2, I'm weighing which layers to drop, with the M9-level features likely mattering most, and planning to test several prefix-ordered configurations in one pass.

I'm weighing a few prefix-ordered layer subsets for the Experiment 2 drop, such as dropping only layer 7 or dropping layers 5 and 7, each giving a different number of real layers at each prefix. Dropping layers usually costs some feature quality, but the later layers may make up for it, and I'd expect the base M10 to land close to M9's 98.2%.

I'm weighing dropping one layer from 1 to 7, which would leave 88.2M params, or two layers for 81.1M. Each exploration pass has a cost, and I have seven passes total, so I'll plan for two exploration passes and keep some spare. For Experiment 2 I'm testing a drop of layers 5 and 7, reading features at the prefix depths after layers 8, 9, and 10 on the 1024-sample subset to compare against the M9 baseline.

I'm checking whether dropping layers 5 and 7 makes sense compared with dropping 3 and 6, using the CKA similarity and relative-change numbers. Layers 5, 6, and 7 show the smallest change, so dropping 5 and 7, which aren't adjacent, looks reasonable, with dropping 6 alone as another option. I'm also considering whether to test only layer 7 in Experiment 2 and leave two-layer drops for Experiment 3.

I'm weighing a two-layer drop of 5 and 7 against a single drop, and the prefix structure means that if the depth-9 cut holds up, it already gives a k=9 fallback without needing a separate run.

I'm checking the carry heads' reserved dimensions to see whether they leak into the layer-norm statistics, which I expect to be negligible. I'll confirm their magnitude before excluding layer 11 and settling on the layer set for the drop test.

I'm stripping the layer edits from Experiment 1, since the output.dense.bias compensation was only meant for that run. The carry heads' reserved dimensions can stay as they are, because the pooler never reads them.



K-means centers reach 98.08% on 5-fold, nearly matching the 98.2% of full kernel ridge, so the pooling layer's FFN can stay off. Next I'll undo the Experiment 1 edits and run layer dropping with layers 5 and 7 removed.



## tool_use: run_python

```json
{
 "code": "for e in list(edits()): drop_edit(e)\nprint('edits now', edits())\nF=np.load(WORK/'base_feats.npz'); RES=np.array([83,88,222,254,444,495,506,601])\nprint('reserved dims magnitude at L9 mean:', np.abs(F['mean_hidden_L9'][:,RES]).mean(), ' other dims', np.abs(F['mean_hidden_L9']).mean())\nn=set_structure(layers=[0,1,2,3,4,6,8,9,10], ffn_keep=3072); print(n)\nwith open(WORK/'notes.md','a') as f: f.write(\"\"\"\n## Exp1 (pass_3): all 12 layers, FFN m=1024 via Gaussian proxy + mean bias compensation -> DESTROYED features (M9 subset CV 97.2 -> 95.0; outlier dim 308 drifts). FFN pruning abandoned. Edits dropped.\n- LOO (closed form KRR) ~ test: original-style head LOO 98.4 vs test 98.31. M9 kernel LOO 98.66 (alpha-mix LNr(5*M9+C9)). M0+M1 LOO 97.9 (shallow too risky). M8 98.06.\n- Linear carry (M2,M6) adds nothing on top of K9.\n- Realizable head: pooler 768 tanh units = exp-kernel features at k-means centers (gamma 0.4, b0~-1), SoR solve (H'H + lam Kcc)B=H'T, lam 1e-3..1e-2: 5-fold 98.08 vs full KRR 98.2.\n- Plan pooling layer (host 11): q=0, V=I, O=alpha*I (non-res), FFN off, LN gamma1/beta0 -> final CLS ~ LN(cls + alpha*mean).\n- Exp2 (pass_4): structure [0,1,2,3,4,6,8,9,10] (drop 5,7) read 1024 ids, check prefixes after 8/9/10 vs base M9 subset (5fold 97.17).\n\"\"\")\n"
}
```

