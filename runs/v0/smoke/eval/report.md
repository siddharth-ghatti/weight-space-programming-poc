# Evaluation: `<repo>/runs/smoke/model`

**Test accuracy 0.8791 | macro-F1 0.8812 | weighted-F1 0.8798 | n=1779**
Practice accuracy 0.9418 | macro-F1 0.9421 | gap +0.0627

## Per-class (test)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| coverage_grant | 0.803 | 0.803 | 0.803 | 233 |
| exclusion | 0.867 | 0.836 | 0.851 | 226 |
| premium_payment | 0.865 | 0.877 | 0.871 | 219 |
| deductible_limits | 0.907 | 0.911 | 0.909 | 225 |
| claims_procedure | 0.945 | 0.928 | 0.936 | 222 |
| definitions | 0.995 | 0.961 | 0.978 | 206 |
| term_cancellation_renewal | 0.905 | 0.857 | 0.880 | 223 |
| conditions_duties | 0.778 | 0.871 | 0.822 | 225 |

## Confusion matrix (rows = gold, cols = pred)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 coverage_grant | 187 | 17 | 4 | 5 | 3 | 0 | 5 | 12 |
| 1 exclusion | 14 | 189 | 0 | 7 | 0 | 1 | 1 | 14 |
| 2 premium_payment | 6 | 2 | 192 | 5 | 5 | 0 | 2 | 7 |
| 3 deductible_limits | 5 | 5 | 6 | 205 | 1 | 0 | 0 | 3 |
| 4 claims_procedure | 1 | 0 | 4 | 1 | 206 | 0 | 5 | 5 |
| 5 definitions | 5 | 0 | 1 | 1 | 1 | 198 | 0 | 0 |
| 6 term_cancellation_renewal | 3 | 2 | 11 | 1 | 0 | 0 | 191 | 15 |
| 7 conditions_duties | 12 | 3 | 4 | 1 | 2 | 0 | 7 | 196 |

## Edited tensors

2 tensors, 6,152 entries changed (0.0056% of the model)

| tensor | changed | fraction | ||delta||_F | ||base||_F |
|---|---|---|---|---|
| classifier.bias | 8 | 1.0000 | 2.358 | 0.000 |
| classifier.weight | 6,144 | 1.0000 | 18.200 | 0.000 |

## Editor's final answer

I left the encoder and pooler unchanged and wrote a closed-form classification head. I ran one forward pass to read the pooled [CLS] outputs for 768 practice sentences (p0000–p0767). I then fit a ridge regression (lambda 0.1) from those pooled vectors to one-hot labels. The result is saved as classifier.weight and classifier.bias, with the bias absorbing the feature mean. No training or iterative fitting was used. On the 168 sentences held out of an earlier fit on the other 600, accuracy was about 87%. The saved head was refit on all 768 sentences, so its 99.9% accuracy on those is a training figure, not a test estimate. I did not use the second forward pass to check the saved model.
