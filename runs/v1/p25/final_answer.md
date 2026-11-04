I turned bert-base-uncased into the 8-label insurance classifier by computing weights in closed form. Nothing was trained: I only read activations through forward passes and solved for weights with ridge regression. Cross-validation on the practice set puts expected accuracy at about 98%. On the 1,024 practice sentences I ran through the final model it got 1,023 right, but those sentences were also used to fit the weights, so that is a correctness check, not an accuracy estimate.

**What I found by reading activations.** I read the hidden states for all 1,496 practice sentences at every layer and compared simple linear classifiers on them with 5-fold cross-validation:
- The average over all of a sentence's tokens works better than the [CLS] token alone.
- The best single layer was layer 9, combining its [CLS] state with its token average (about 97.6%).
- Adding the average of the original word embeddings (layer 0) raised this to about 98.1%.

**What I changed.**
1. **Layer 0 "stash":** one attention head now attends equally to every token, so it computes a sentence average. It writes a small learned summary of the embedding average (8 numbers, one per label) into 8 hidden dimensions that were otherwise little used.
   - In layers 0–8, reads from those 8 dimensions are replaced by their average value, writes into them are blocked, and their LayerNorm settings are fixed. The rest of the network runs essentially as before, and the summary reaches layer 9 intact.
   - I confirmed this with forward passes: the stashed values reappear exactly at layer 9, and accuracy is not hurt.
2. **Readout at layer 9:** an equal-attention head adds the sentence average of the layer-9 states into the [CLS] position as 8 label-score channels.
3. **Final-layer weights:** the pooler is set up to compute "fixed scores from the [CLS] state + those channels". Its tanh preserves which score is largest, and the classifier just copies pooler outputs 0–7 to the 8 labels.
4. **Layers 10 and 11 pass [CLS] straight through.** Their attention output and feed-forward weights are zeroed and their LayerNorm settings made neutral.

The label weights come from one ridge regression on the layer-9 [CLS] state, the layer-9 average and the stash. I chose the regularisation strength (500) by repeated cross-validation.

**Checks.**
- On all 1,024 sentences I ran (two batches of 512, including the longest ones), the model's logits equal my offline calculation times a positive per-sentence factor from LayerNorm. Per-sentence correlation is above 0.9999 and the predicted labels always agree.
- The single error was p1237, an "assignment" clause (conditions_duties) predicted as claims_procedure.
- The largest logit is 0.46, so the tanh is not saturated and there are no ties.

A nonlinear readout scored only about 0.1% higher in cross-validation, which is within noise, so I kept the simpler linear version.