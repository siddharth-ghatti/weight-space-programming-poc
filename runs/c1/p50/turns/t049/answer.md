## reasoning summary

I'm finding that even the deep-layer mean features, with the leaky dimensions removed, reach about 97.5% with the kernel and 96.6% with linear. The original model got 98.31% on test by combining layers 2, 6, and 10 with the final CLS, so deeper contextual features don't add much beyond the layer-0 bag-of-embeddings kernel at around 97.1–97.9%. I'm also considering a lower gamma, since gamma 2 did better than 5, and might try 0.

I'm weighing whether a single-layer model, built on the layer-0 embedding features with a good kernel, could reach roughly 97.7–98.0% on its own, since the information ceiling for any single-layer mean feature looks to be around 98%. Attention-weighted pools such as max-style or positional pooling might add more. Computing those pools over all 1496 sentences would mean rebuilding the embedding layer in numpy, which is a gray area I still need to sort out.

I'm weighing whether to avoid computing the embedding layer in numpy and instead read features from forward passes of a designed one-layer attention model, accepting roughly two passes per 1496 sentences and about 20 design iterations.

I'm checking whether the top PCA directions of the sentence means carry most of the signal, so the kernel could run on fewer dimensions. I'm also weighing an IDF-style pooling, where a learned key direction would score each token's content weight before averaging.

I'm considering whether a supervised attention key, fit on one half of the data, could weight tokens for the other half without leaking, so the pooled features stay honest.

I'm weighing whether to estimate the value of attention-weighted pooling without leakage, perhaps by restricting attention keys to a low-dimensional class-evidence subspace under strong ridge regularization. I'm also thinking about which classes get confused, likely exclusion versus coverage, since that would show where pooled features could help. The token-level layer-0 states I have only cover 64 sentences, so a cheap check on those is limited.

I'm checking whether combining the layer-0 kernel with deeper mean features could push past 98.2%, since the original model's gain came from merging several layers with the CLS vector. I'm also considering a two-layer design with the first layer built to carry contextual information forward.

I'm wondering whether a single layer of attention pooling could reach the accuracy I need. The estimates are noisy, and with a test set of a thousand or so, a true accuracy near 98% still leaves a real chance of landing below the target, so I want the highest honest number I can get.

I'm planning to split the practice sentences into two halves, fitting all the supervised attention components on one half only. The other half would be read out as features and scored with 10-fold cross-validation, so the estimate stays honest.

I'm considering how to build honest supervised features without token-level data, and I realize the class-evidence weights were fit on sentence means. Since a linear model on the mean embedding is equivalent to averaging token-level linear scores, fitting those weights on one half of the practice set keeps the evaluation on the other half clean.

I'm now going to test which pooled features carry the useful signal, since position-based and weighted pools might help where the plain mean stalls. The cheapest way is one forward pass that reads a bank of candidate pooled features, so I can compare them without rebuilding the model each time.

I'm checking whether projecting the mean features onto their top PCA directions keeps the accuracy, which would let the kernel run on fewer dimensions. I'm also planning a few unsupervised pooling designs, with position-weighted and IDF-style attention heads, and I'll compare them against the mean features in a kernel CV.

I'm checking whether the mean features need all their dimensions, or whether a few heads could carry the mean while the rest stay free for other pools. I also want to compare kernel gammas of 1 and 0.5, use 20-fold CV, and test whitening or standardization.



Single-layer mean features hit about 97.5% with a kernel at every depth, and raw embeddings do nearly as well, so deep layers add little. I'm next checking how many mean-embedding dimensions the kernel needs and tuning its scale with 20-fold CV.



## tool_use: run_python

```json
{
 "code": "exec(open('lib.py').read())\ny=labels(); B=load_base()\nM0=B['mean_hidden_L0'].astype(np.float64)\nXc=M0-M0.mean(0)\nu,s,vt=np.linalg.svd(Xc,full_matrices=False)\nev=s**2/np.sum(s**2); print('cum var', [round(ev[:k].sum(),3) for k in [8,32,64,128,256,384]])\ndef kcv(Z, gs=(0.5,1,2,3), lams=(0.003,0.01,0.03), k=10):\n    Z=Z/np.linalg.norm(Z,axis=1,keepdims=True); S=Z@Z.T\n    return {g:krr_cv(np.exp(g*(S-1)),y,lams=lams,k=k,seeds=(0,)) for g in gs}\nprint('full', kcv(Xc))\nfor kd in [64,128,256,384]:\n    Z=Xc@vt[:kd].T\n    print(kd, kcv(Z,gs=(1,2),lams=(0.003,0.01)))\n# whitened-ish: scale pcs by s^-a\nfor a in [0.25,0.5]:\n    Z=(Xc@vt[:384].T)/s[:384]**a\n    print('white',a, kcv(Z,gs=(1,2,4),lams=(0.003,0.01,0.03)))\n"
}
```

