I turned BERT into a keyword-scoring (bag-of-wordpieces) classifier that I built by hand. With no forward passes, it was never actually run, so test accuracy is untested. The table of word scores it uses gets about 97.2% accuracy in 5-fold cross-validation on the practice set (per-class F1 0.957–0.982), and the network should match it closely. Every edited tensor was checked piece by piece against what it is supposed to do.

**How the model classifies:** each WordPiece token in the sentence gets an 8-number score, one per label. The model averages these over the sentence and predicts the label with the highest average.

**Where the scores come from:** a blend of two quick statistical models fitted directly on all 1,496 practice sentences, with no iterative training:
- naive-Bayes word statistics (smoothing 0.3), weighted by ¼;
- ridge regression on word frequencies (regularization 3e-3).

The blend scored slightly better in cross-validation than either alone (NB ≈96.0%, ridge ≈97.1%). The special tokens ([CLS], [SEP], [PAD], [UNK]) and words never seen in practice get a score of 0.

**What I edited:**
1. **Layer 0:** attention is switched off (its output weights are zero). Its LayerNorm zeroes 8 residual dimensions; these become class slots, with dimension DF[c] holding the score for label c. The feed-forward layer is a bank of 3,072 keyword detectors, one per informative practice token. Each one recognises its word from the token's input embedding, in a way that ignores the word's position. Their output weights were solved in one step so each token writes its 8 scores (scaled by 0.3) into the class slots, at any position.
2. **Layer 1:** query and key weights are zero, so attention spreads evenly over all real tokens. The value weights copy the class slots, so [CLS] receives the sentence average, multiplied by 4. Padding is masked out; even without a mask, padding scores 0 and can't change the prediction.
3. **Layers 2–11:** exact no-ops. Their attention and feed-forward outputs are zero and their LayerNorms are plain normalisations, which leave the result unchanged.
4. **Pooler and classifier:** the pooler reads the 8 class slots at scale 0.5, staying in tanh's near-linear range. The classifier is an identity on those 8 values with zero bias, so the prediction is the label with the highest average score.

**What I checked (from the weights alone, without running the model):**
- The detectors reproduce the target scores within about 1–2% at normal positions and about 5% at position 1. Direction stays right (cosine ≥ 0.995) even at position 127.
- Special tokens output about 0 at every position up to 511.
- Unseen words contribute very little, and where they do it points the sensible way (e.g. "canceled" → cancellation, "purposely" → exclusion).
- All 108 edited tensors have the right shapes and contain no NaNs or infinities.

**What I left out:** word pairs (bigrams) would add only about 0.3–0.7% in cross-validation. They would need a hand-built "look at the previous word" attention head that I couldn't test with zero forward passes, so I didn't add one.