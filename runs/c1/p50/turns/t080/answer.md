## reasoning summary

I'm comparing the results across the 1496-sample runs: M9 alone is at 98.4, and adding M0 or C9 pushes it to 98.53, so M0's gain is small. On the 1024-sample set, M0 helped more, and M9 alone scored 97.61, which suggests the sample size itself is making a big difference, possibly because the extra 472 sentences are easier.

I'm checking whether the drop from 98.4 to 97.61 comes from the training size or from the subset itself, so I want to rerun the 1496-trained model and score it only on the first 1024 ids. The final model will train on all 1496, so my 10-fold estimate on 1346 training samples is a bit pessimistic. For pruned models I'll read all 1496 with two passes per evaluation. On the 1024 ids, pruning 1024 units drops M9 alone from 97.61 to 97.07 and M0+M9 from 98.44 to 98.10. I'm thinking the next step is a sequential refit that matches each pruned layer's output to the base model's output on the same tokens, so upstream errors don't keep compounding.

I'm weighing whether a sequential refit that matches each pruned layer's output to the base model's output on the same tokens is worth the extra passes. It would cost about nine more passes plus the later evaluation and verification, which still leaves some headroom. The FFN can only partly undo upstream error, though, and the gain from m=1536 over m=1024 is about 11% in parameters, so I'm not sure the extra work pays off yet.

I'm asking what accuracy I actually need: the test set has to reach at least 97.8%, and my 10-fold CV on the 1496 samples should be a fair guide to that. So I'm aiming for a CV score of about 98.3%. The base M9 model alone is at 98.4, but the pruned version at m=1024 probably lands around 97.9 to 98.1, which falls short. I need either a wider m=1536 or a sequential refit, or a better set of features.

I'm considering whether combining the M5 or M7 layers with M9 could beat M9 alone, which would be worth checking before I commit to carrying information forward. The carry heads in layers 2 and 6 were important in the original design, so I'm thinking about refitting them to write their scores into the reserved dimensions.

I'm checking whether the carry score can be simulated exactly in numpy. Since the carry head's uniform attention makes its output linear in the layer-l mean, I could compute it from the stored readouts for honest cross-fitting, provided the mean includes CLS and SEP.

I'm working out how to validate the carry channel honestly: the carry scores are linear in the layer-l mean, so I can compute them per fold, but a final pooler trained on in-sample carry scores would over-trust them, so I need cross-fitting.

I'm wondering whether layering in earlier-layer signals could help, so I'll test kernel CV on M9 combined with M0, M2, or M5 as a quick check. If the gain is under about 0.1%, it isn't worth the extra complexity.



M9 alone reaches 98.40% on all 1496 sentences, and adding M0 brings it to 98.53%, so the earlier gap on the 1024 subset was mostly a training-size effect. I'm now checking whether an earlier layer's mean combined with M9 is worth carrying forward, along with a 20-fold reference for M9.



## tool_use: run_python

```json
{
 "code": "exec(open('lib.py').read())\ny=labels(); B=load_base()\ndef cen(X): X=X[:,KEEP].astype(np.float64); return X-X.mean(0)\ndef bn(X): return X/np.sqrt((X**2).sum(1).mean())\ndef kcv(Z, gs=(0.5,1), lams=(0.003,0.01,0.03), k=10, seeds=(0,)):\n    Z=Z/np.linalg.norm(Z,axis=1,keepdims=True); S=Z@Z.T\n    return max(max(krr_cv(np.exp(g*(S-1)),y,lams=lams,k=k,seeds=seeds).values()) for g in gs)\nM={l:bn(cen(B[f'mean_hidden_L{l}'])) for l in [0,2,4,5,6,7,8,9]}\nprint('M9 2seeds', kcv(M[9],seeds=(0,1)))\nfor l in [2,4,6,7,8]:\n    print('M%d+M9'%l, kcv(np.concatenate([M[l],M[9]],1),seeds=(0,1)))\nprint('M9 20f', kcv(M[9],k=20))\n"
}
```

