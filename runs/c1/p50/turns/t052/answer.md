## reasoning summary

I'm checking whether the KEEP dimensions at layer 12 leak information. Layer 11 was rewritten to pool, and the reserved channel passes through LayerNorm, whose mean and variance computation includes those reserved dimensions. So the KEEP dims of layer 12 may carry some leakage through those LN statistics.

I'm also wondering whether the layer 11 attention might route information from the reserved dimensions into the KEEP dims, and whether the final LayerNorm's mean and scale over all 768 dimensions leak a small amount of label signal. The leakage through those scalar statistics looks tiny, so it probably doesn't account for the roughly 0.6% gain from deeper features. The open question is whether a single layer with attention-weighted pooling could reach about 98.3%.

Since one layer with smart pooling is the only option left, I'm now designing an experiment that uses class-evidence-keyed attention pools, where each head attends to tokens scoring high for a given class, and I'll cross-fit the halves to evaluate it honestly.

I'm weighing a design where some heads pool uniformly over tokens after PCA projection, since the mean of projected tokens equals the projection of the mean. The other heads would be class-evidence-keyed, using ridge directions fit on one half for an honest evaluation. The value vectors would combine class-score dimensions with top PCA components. The open concern is whether a token-level scorer built from sentence-level ridge weights can be trusted to pick out the right tokens.

I'm weighing a cross-fitted design where the class-evidence directions are fit on one half and the features are evaluated on the other, with a paired comparison against the baseline on the same held-out half to keep the noise down.

I'm working out how to avoid the classifier learning from leaked features: if the attention keys are fit on the full data, the class-score values look better on training than they will at test time. One fix I'm considering is training the classifier on cross-fitted features instead, so it sees the same kind of signal it gets at test time.

I'm weighing whether to skip the forward passes and instead fit the supervised keys on one half and read only the other, so each design gets evaluated honestly with a single pass. The worry is that the classifier trained on those features will over-trust the class-score values, so stacking cross-fitted features from both halves may be the safer route.

I'm going to check the leaky upper bound first: I'll evaluate the E1 unsupervised features on top of M0 and see whether they beat the 97.8% baseline even with leakage. If they don't, I'll drop the idea.



Adding the layer-12 mean to M0 lifts kernel CV from about 97.7% to 98.4%. Next I'll test whether E1's keyed-pool features help M0 even with leakage, as a quick upper-bound check.



## tool_use: run_python

```json
{
 "code": "exec(open('lib.py').read())\ny=labels(); B=load_base()\nU=np.load('E1_u.npy').astype(np.float64)\nL='bert.encoder.layer.11.'\now=W(L+'attention.output.dense.weight'); perm=np.argmax(np.abs(ow),axis=0)\ndef bn(X): X=X-X.mean(0); return X/np.sqrt((X**2).sum(1).mean())\ndef kcv(Z, gs=(0.5,1,2), lams=(0.003,0.01,0.03), k=10):\n    Z=Z/np.linalg.norm(Z,axis=1,keepdims=True); S=Z@Z.T\n    return {g:max(krr_cv(np.exp(g*(S-1)),y,lams=lams,k=k,seeds=(0,)).values()) for g in gs}\nM0=bn(B['mean_hidden_L0'].astype(np.float64))\ncls_dims=np.concatenate([perm[64*h:64*h+8] for h in range(12)])\nrest=np.setdiff1d(np.arange(768),cls_dims)\nprint('U', kcv(bn(U)))\nprint('M0+Urest', kcv(np.concatenate([M0,bn(U[:,rest])],1)))\nprint('M0+U', kcv(np.concatenate([M0,bn(U)],1)))\n# per head pools PCA parts\nfor h in [8,9,10,11]:\n    d=perm[64*h+8:64*h+64]\n    print('M0+head',h, kcv(np.concatenate([M0,0.5*bn(U[:,d])],1),gs=(1,)))\n"
}
```

