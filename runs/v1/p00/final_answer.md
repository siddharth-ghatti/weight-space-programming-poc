I turned BERT into a classifier that averages the sentence's input embeddings and applies a linear model, built entirely by writing weights. The forward-pass budget was 0, so I never ran the model and none of this was checked on the real network. The expected accuracy comes only from cross-validation on stand-in features, not from model outputs: about 95.8% accuracy and macro-F1 (5 seeds × 5 folds). If something in the construction doesn't match what the harness actually runs, the score could be much lower.

**How the edited model works (106 tensors edited)**
1. **Layer 0 attention averages the sentence.** The query and key weights and biases are set to zero, so every token attends equally to all real tokens, including [CLS] and [SEP]. Every position, including [CLS], therefore receives the average of the value vectors over the sentence.
2. **The value layer computes 8 class scores.** Its first 8 rows are G = A·diag(1/γ_emb), and its bias is −A(β_emb/γ_emb). This undoes the frozen embedding LayerNorm's scale and shift, so the averaged value equals z = A·mean_j(n_j), where n_j is token j's normalised embedding. A is built to ignore three things that don't depend on the word:
   - the all-ones direction, which LayerNorm subtracts anyway;
   - the token-type-0 embedding;
   - the top 30 directions of the position embeddings.
   Without that position projection, position information flips 45 of 1,496 predictions; with it, only 1.
3. **Writing the scores in.** The attention output layer writes 10⁴·Q·z into the residual, where Q is 8 orthonormal zero-mean directions. Because this term is so large, the [CLS] token's own embedding becomes negligible after the LayerNorm. The worst-case shift is 0.003, against a 1st-percentile margin of 0.06.
4. **Everything else passes the signal through unchanged.** In all layers the feed-forward output weights and biases are zero, and in layers 1–11 the attention output weights are zero too. Every LayerNorm is set to weight 1 and bias 0. The normalised [CLS] vector therefore reaches the end of layer 12 unchanged.
5. **Readout.** The pooler reads the 8 Q directions at a small scale (0.02), and the classifier weight is [I₈ | 0] with zero bias. Since tanh is increasing, the prediction is simply argmax(z).

**How A was fitted (closed form, practice data only)**
- I used ridge regression with no intercept (λ = 0.3) on centred one-hot labels. With no intercept, the argmax doesn't change if the real features differ from mine by an overall scale.
- I didn't compute the embedding-layer outputs, because that would mean re-running part of the model. Each sentence's features are the projected average of its raw word-embedding rows, with each token divided by an estimate of its LayerNorm spread built from weight norms.
- The result barely changes however that spread is modelled (fitting with one version and testing with another changes accuracy by under 0.5%).
- I also tried weighting rare tokens more heavily when averaging (IDF weighting). It made accuracy worse, because words like "not", "must" and "means" carry class signal, so I kept the plain average.

**Assumptions I couldn't check**
- The model runs in fp32 or bf16. The scale of 10⁴ also stays inside fp16's range.
- Padding is hidden with an attention mask.
- Sentences are single segments with token type 0.

The remaining cross-validation errors look like genuinely borderline sentences, for example "notify us promptly…", which sits between claims procedure and conditions.