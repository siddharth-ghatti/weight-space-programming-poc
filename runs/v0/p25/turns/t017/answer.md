## reasoning summary

The final CLS matches my prediction to within 1e-5, so the mean over hidden states covers every token, the same as uniform attention. Cross-validation on the pooled features gives 99.6%, but that number is leaky since M was designed using all the labels, so the honest estimate is around 97.5%. Now I'm weighing how to pick the final head, starting with a direct linear map on the reconstructed representation.

I'm weighing whether to fit the head on the pooled labels or on the designed logits, and the label fit would mostly just reuse the direction M already separates on. Any honest estimate needs a design split that excludes some held-out folds, which means redoing the reads rather than reusing the current ones.

I'm checking whether running the normalization arithmetic myself counts as reimplementing the forward pass, and I'm leaning toward keeping evaluation to the linear-algebra CV I already did, since the normalization effect is tiny. The refit folds came in at 0.973 to 0.976, which is close enough to the direct estimate of 0.975 that I'm not worried about the head choice. Next I want to see whether combining the first, middle, and final CLS reads can push past the 0.983 ceiling.

I'm thinking about whether layer 10's output could carry layer-0 information forward, so that layer 11's pooling can see per-token embedding features. A reserved-dimensions approach or a CLS accumulator might let me get mean0 into the final read without a separate pass.

I'm considering whether to route the layer-0 information forward through the CLS token itself, accepting some perturbation to other tokens since the final features get refit anyway. The idea is that CLS in early layers gets attended to, so its representation could carry mean0 into layer 11 without a separate pass.

I'm checking whether I can make layer 0 behave differently for the CLS token alone, since its input is the same constant vector for every sentence. The trouble is that other tokens' embeddings share components with that vector, so tweaking the query weights for CLS would also nudge the rest of the sequence.

I'm looking for a readout direction that is large on the CLS embedding and near zero on every other token's embedding, since CLS is the only token at position 0. Other tokens' embeddings overlap with that vector, so a clean separation may not exist, but a few typical tokens could still show whether it holds.

I'm considering zeroing out one layer-0 head so that its query is constant, which would make it attend uniformly, but other tokens would still pick up that head's output through the value path. I need the head's contribution to be near zero for non-CLS tokens, and the open question is whether a readout that isolates the CLS embedding exists at all.

I'm working out whether the value path can be made to carry the CLS-only signal: setting the value bias to cancel CLS's own value would zero its contribution, leaving the other tokens' values as shifted differences from the CLS embedding, which should still be informative.

I'm sketching a single query-key dimension in one layer-0 head that scores high for non-CLS queries against the CLS key, so those tokens send their attention entirely to a zero-valued CLS and output nothing. The CLS query stays constant and attends uniformly, which would give it the mean over the other tokens' values.

I'm working out how to make the head respond only to the CLS token: a single query-key dimension that gives non-CLS queries a strong pull toward CLS, with CLS's own query kept near zero so it attends uniformly. The remaining difficulty is that the non-CLS tokens' readout has to sit very close to zero.

I'm setting the query and key weights so that one dimension of the head pulls non-CLS tokens strongly toward the CLS key, while the CLS query stays near zero and attends uniformly. Then I'm checking whether a readout direction can keep the non-CLS tokens' values close to zero, since the embedding outputs would have to be computed to verify it.

I'm trying to find a readout direction r where r·e_CLS equals 1 and r·x stays near zero for every other token. Averaging over a sentence doesn't pin that down, since it only gives one constraint per text, so I'd need per-token embeddings to fit it properly. Reading the layer-0 hidden states directly from the tool would supply those.

I'm working out whether a linear readout can isolate the CLS embedding. Since layer normalization divides each token by its own standard deviation, the zero-on-other-tokens condition only holds cleanly if the readout is orthogonal to the bias term and to the gain-weighted all-ones direction, so I'm checking what that requires of the gain-scaled readout.

I'm weighing whether a readout that isolates the CLS embedding can work at all, since it would need a direction that is zero on every other word and position. Making that hold for all 30,000+ words is impossible, so I'd only need it approximately for typical tokens. The position embedding for position 0 might be separable from the nearby positions, but positions are likely smooth and low-rank, so that separation may fail for longer inputs.

I'm weighing whether the readout can be made exact by fitting a direction that is orthogonal to every word and position embedding except the CLS one. Least squares over the vocabulary could push the word components toward zero, though I'd still need the layer-norm step to see whether that holds.

I'm checking whether the embedding-layer condition can be met purely through linear algebra on the weights, without running any forward pass. If the readout direction is built so that its gain-weighted version is orthogonal to the word and position vectors, the normalization step should cancel out cleanly. Then I'm looking at how the CLS-only readout would need to be constructed from there.

I'm noting that a dedicated head is needed, since a shared head would break its original role for every token. That means accepting a loss of one layer-0 head's function, and then looking at how the CLS representation passes forward through the layer-1 onward stack.

I'm checking whether the bag-of-words information from layer 0 survives through layers 1 to 10 into the CLS output at layer 11. Since the CLS token's attention in those layers may behave oddly, that's uncertain and needs a test, so I'd read the layer-11 CLS and mean hidden states for the new model.

I'm wondering whether a dedicated layer-0 head is worth it, and whether to sacrifice the head with the smallest output norm. Pooling at a later layer might be cleaner, but I'd rather check the simpler things first.

I'm going to save the v1 classifier first so there's a working model to fall back on, then run a quick predict check on a sample. After that I'll look at the layer-0 CLS accumulator idea, which seems like the most promising bigger change.

I'm now trying to make the classifier's logits linear in the pooled features. Since layer norm divides by a per-token standard deviation that barely varies, I can approximate the logits as a constant times a projection of the layer-norm output, and then invert the tanh to recover that output from the pooled vector.

I'm now weighing whether the simpler route is enough: fit a ridge regression from the pooled vector to the designed logits, computed from the base features, which would mirror the designed classifier. Honest cross-validation put that at about 0.975, so I'm comparing it against refitting directly to the labels, which may do better or worse.

I'm weighing whether to compare the label-refit ridge against the designed direct classifier on the practice data, checking whether their predictions agree at a rate above 99.5%. Since a nested honest CV is hard to set up with the layer-norm and tanh steps, I'm leaning toward a ridge refit on the pooled features with a moderate penalty of 30, since the CV results for every option land within a narrow band.

I'm saving the v1 classifier now so there's a working model to fall back on. Then I'll run a predict pass on 512 practice items to check that the harness behaves as expected, which should land near 99.5% since that's training data.



The edited final CLS matches the predicted LN(cls11 + M·mean11) to within 2e-5, so the mechanism works as designed. The 99.6% CV is optimistic because M used all labels; the honest fold-wise estimate is about 97.5%. Next I'll save a v1 ridge classifier on the pooled outputs and check it with a predict pass.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\ny=np.load('base_feats.npz')['y']\npooled=np.load('v1_pooled.npy').astype(np.float64)\nWt,b=ridge_fit(pooled,y,30)\npred=np.argmax(pooled@Wt.T+b,1); print('train acc', (pred==y).mean())\n# compare with designed direct logits\nD=np.load('base_feats.npz'); c11=D['cls11'].astype(np.float64); m11=D['mean11'].astype(np.float64)\nWc,bc=ridge_fit(np.hstack([c11,m11]),y,100)\npd_=np.argmax(np.hstack([c11,m11])@Wc.T+bc,1); print('agreement refit vs designed', (pd_==pred).mean())\nsave_edit('classifier.weight', Wt.astype(np.float32)); save_edit('classifier.bias', b.astype(np.float32))\nnp.save('v1_cls_W.npy',Wt.astype(np.float32)); np.save('v1_cls_b.npy',b.astype(np.float32))\nnp.save('v1_expected_pred.npy',pred)\nwith open('notes.md','a') as f: f.write(\"\"\"- v1 verified: edited final CLS == predicted LN(cls11+M mean11) (max err 2e-5). Reads in pass_3/4 (pooled in v1_pooled.npy).\n  Classifier = ridge(lam30) on pooled; train acc ~1.0; honest CV estimate ~0.975. Saved (v1_cls_W/b.npy).\n- Idea for more: layer-0 dedicated head that pools bag-of-embeddings into CLS only (CLS id via vector r with r.e_CLS=1, r.x~0 others), so cls11 carries mean0 info (ceiling 0.983).\n\"\"\")"
}
```

