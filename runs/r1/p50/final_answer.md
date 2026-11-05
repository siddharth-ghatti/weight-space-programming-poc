I kept all 6 encoder layers and cut every feed-forward block from 3072 to 1536 units (m=1536). The model has 68,136,964 parameters, 17.2% fewer than the original 82.3M (32.8% fewer outside the frozen embeddings). On the 1000 practice paragraphs it agrees with the original model's predictions 98.0% of the time, with accuracy 0.909 and macro-F1 0.897 against the original's 0.925 and 0.914. My estimate is still about a 13–14% chance (range roughly 10–23%) of falling below the test accuracy or F1 bar.

**What I changed**
- **Which units to keep.** For each layer I fed practice paragraphs through the model and recorded the exact inputs to the feed-forward block, token by token. In layers 0–2 I kept the 1536 units that contribute the most output variance. In layers 3–4 I ranked units by their effect after the following LayerNorm, since that is where their output actually goes. In layer 5 only the `<s>` token feeds the classifier, so I ranked using that token alone.
- **Compensating for the dropped units.** I refit each layer's output-projection weights and bias with a closed-form ridge regression (ridge strength 0.1, scaled to the data). The kept units' activations predict whatever the dropped units used to contribute, so the dropped part is folded into the remaining weights. The fitting data came partly from the original model and partly from the already-pruned model, so later layers were fitted on the inputs they actually receive. Only `output.dense.weight` and `output.dense.bias` of each layer were edited; the classifier head is unchanged. There was no training or iterative fitting.

**Why m=1536**
- **Alternatives tried:** dropping a whole layer, narrowing to 1024 units, refitting the classifier head, a greedy unit-selection method, and linear corrections to the logits. All were worse or made no difference.
- **What limits the method:** the remaining error behaves like noise. It is larger on short paragraphs and on paragraphs where the original model is already unsure. It is no smaller on the paragraphs used for fitting, so it should carry over to the test set.
- **How I estimated risk:** I simulated many 400-item test sets. Each one shrinks the original model's practice logits until its accuracy is about 77.5%, then adds the logit noise each pruned version actually showed on practice items with similar margins. I counted a failure if more than 4 net items are lost or macro-F1 drops by more than 0.02.
- **Comparison of sizes:**

| Units per layer | Size reduction | Estimated fail chance |
|---|---|---|
| 1536 | 17.2% | ~13–14% (range ~10–23%) |
| 1792 | 14.3% | ~11–13% |
| 2048 | 11.5% | ~7% |

  If you treat "size reduction if it passes" as the payoff, 1536 gives the best expected result. Extrapolating to 1280 units gives about the same expected payoff with more risk.
- **Which of three near-identical 1536-unit builds:** I submitted the second build ("v2"). Its risk estimate matched the greedy-selection build (~13.5%) and was below the first build's (16.4%), and it held up better than the greedy build when I artificially shrank the margins.

I confirmed the exact submitted edits and structure.json with a final forward pass (pass_32); they reproduce the earlier evaluation exactly.