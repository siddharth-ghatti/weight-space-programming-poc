## reasoning summary

The mean0 and mean11 combination reaches 97.8%, slightly better than the 96.9% baseline and the 97.5% combos. Routing both into layer 11 looks hard, since attention there only sees layer-11 inputs, though some embedding information might survive in certain tokens. I'll check the confusion matrix next to see which classes get mixed up.

I'm planning to edit layer 11 for mean pooling: zeroing the query weights for uniform attention, setting the value weights to identity, and scaling the output projection. The open question is whether I can keep the CLS signal alongside the mean in separate subspaces, like the concatenation that scored 0.974, rather than summing them. Projecting each onto a few discriminative dimensions might work, but it's more involved than the small gain justifies.

I'm working out whether the layer-11 edit can be made exact in the linear case: if the final logits are A·h_CLS + B·mean, the residual path can carry h_CLS through unchanged while a matrix M applied to the mean supplies the B term. Choosing M as the minimum-norm solution of A·M = B would make that work, with layer normalization handled by a large constant bias.

I'm checking whether the layer-11 edit stays exact after layer normalization: the centering is linear, but the standard deviation term only stays negligible if the constant offset K is much larger than the typical size of the pooled vector.

I'm working through whether the linear approximation holds after layer normalization: the centering is exact, and a large constant offset keeps the standard deviation term close to uniform, so the logits only scale slightly and the argmax should hold. The classifier weights can be kept orthogonal to the offset direction so the constant part doesn't drift with the scale. Next I'm checking the rest of the path, where the residual goes through the feed-forward block (switched off) and the second layer norm, which is already close to identity, and then the pooler with its tanh.

I'm considering a simpler route: apply LN directly to h_CLS plus M times the mean, and read off the pooled output to fit LDA. A scaled identity for M gives roughly the sum variant at about 0.969. The better option is to pick M from the concatenation fit, so the combined discriminant signal lands in the final features.

I'm weighing whether to refit the classifier on the final features or use the designed weights directly, since the concatenation target is already known. Refitting on about 1500 samples should match what cross-validation measured, so that's fine. Still, a half-point gain is small, so I want to check the confusion matrix first to see whether the remaining errors are real ambiguity or label noise.

I'm checking the misclassified examples for the mean11 LDA, and considering whether mean0 and mean11 could be routed together. Layer-11 attention only sees layer-11 inputs, though the residual stream may still keep enough token identity from the embeddings to make that workable.

I'm weighing whether a uniform-attention head in layer 11 could write the mean into the CLS residual, though that would change what other tokens read from CLS and might be risky. Simplest robust option seems to be uniform attention in layer 11 while still keeping h_CLS, and I'll check the errors before going further.



Combining features adds only about 0.5% (97.3–97.8% vs 96.9%). Next I'll check the mean11 LDA confusion matrix and misclassified sentences to see whether the errors are fixable or just ambiguous labels.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nD=np.load('base_feats.npz'); y=D['y']; folds=np.load('folds.npy')\nP=practice()\nacc,pred=cv(D['mean11'].astype(np.float64),y,folds,lda_fit,0.1)\nprint(acc, macro_f1(y,pred))\nC=np.zeros((8,8),int)\nfor a,b in zip(y,pred): C[a,b]+=1\nprint([l[:6] for l in LABELS]); print(C)\nfor i in np.where(pred!=y)[0][:45]:\n    print(LABELS[y[i]][:10],'->',LABELS[pred[i]][:10],'|',P[i]['text'][:160])"
}
```

