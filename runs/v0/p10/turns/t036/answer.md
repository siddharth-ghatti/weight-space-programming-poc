## reasoning summary

I'm going to wrap up v1 now. Before submitting, I'll check that the edits are still in place and that the classifier was saved, then update the notes. Another predict pass on a different sample doesn't seem necessary, since it's all training data anyway. I'm also reconsidering whether to use the remaining passes for augmentation with hand-written sentences.

I'm weighing whether more labeled data would help. The learning curve is still rising, but rephrased or templated sentences would mostly act as regularization and add little. Hand-written prototypical examples would add little too, since the errors are on borderline cases. So I'll skip augmentation and finalize v1, updating the notes first.



Length features don't explain the layer-0 gain, so I'm stopping with v1, which is verified and near the best simple design. I'm checking that the saved edit set is complete and then updating the notes.



## tool_use: run_python

```json
{
 "code": "\ne = edits(); print(len(e)); print(sorted(e)[:3], '...')\nWc = W('classifier.weight'); print(Wc.shape, np.abs(Wc).sum()>0, W('classifier.bias'))\nopen('notes.md','a').write(\"\"\"\n## Verification + final (pass 6 predict edited n=512 seed1): logits match expected to 5e-5; practice acc 99.4% (train-fit).\n- 10-fold CV of final head on v1 features: 97.2%. Explored extra streams (cls9+mean9 routing ~+0.15%, residual mixing no gain, length no gain) -> not worth it.\n- edits_v1_backup/ holds the final edit set. FINAL = v1. Submitting.\n\"\"\")\n"
}
```

