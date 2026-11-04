

## turn 1

Strategy: run the encoder for real (read passes) to get activations, then fit the classifier with closed-form math in the sandbox. Steps:
1. Read base hidden states (CLS and mean-pooled, all layers) for all 1,496 practice sentences. That takes 2 read passes because each pass holds at most 1,024.
2. Use cross-validated closed-form classifiers (ridge/LDA) to find which representation separates the 8 labels best.
3. Edit the last encoder layer(s) so the [CLS] output carries that representation, for example by making attention uniform so [CLS] becomes a mean-pool. Then read the edited model's actual `pooled` outputs and fit the classifier head in closed form on them.
4. Check the result with a predict pass.

First I'll look at the data and set up notes.

## turn 2

Now reading base CLS and mean-pooled hidden states at all 13 layers for the first 1,024 practice items (pass 1 of 2 for the base features).

## turn 4

The ids are sequential. Now the real base read for p0000–p1023: CLS and mean hidden states at all layers.

## turn 5

Pass 2 worked. Before using another pass, I'll run 5-fold cross-validation with closed-form ridge/LDA on these 1,024 items to see which layer and pooling (CLS vs mean) gives the best features.

## turn 18

v1 edits are saved (38 tensors). Next, a read of the edited model's `pooled` output (plus final [CLS] hidden) for p0000–p1023.

## turn 19

Pass 4 done. I'll confirm the edited final [CLS] equals the normalised layer-8 mean, then cross-validate ridge on the actual `pooled` outputs.

## turn 27

The classifier is saved. Next, a predict pass on the edited model to confirm its logits match the ones I computed from the pooled reads. That checks the whole pipeline end to end.