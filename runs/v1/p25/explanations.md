

## turn 1

I'll start by checking whether there are notes from an earlier session, then look at the data and the helper environment.

## turn 4

That one-text read (pass 1) is meant to pin down how `mean_hidden` is defined, i.e. whether [CLS]/[SEP] are included in the mean.

## turn 8

Indexing bug: K already has only the training columns. Fixing it.

## turn 19

Pass 6 has the attention maps (attention_L1 = encoder layer 0). Next I'll compute how much attention mass non-[CLS] tokens put on [CLS], per layer and head.

## turn 36

Verifying the layer-9 readout with one predict pass, then checking exactness against the offline logits.

## turn 41

Let me look at the working folder state to re-orient (files, readout.py helpers) before deciding whether to pursue the nonlinear readout plan.

## turn 43

The kernel result is sensitive to β/θ. Next I'll check the distribution of cosine similarities and run a finer grid with repeated CV over several fold seeds, so noise doesn't decide the choice.

## turn 47

I'll measure how close each layer-0 head (attention_L1 = first encoder layer) is to uniform attention, to pick the cheapest head to sacrifice for M0 pooling.

## turn 81

I'll check the saved state first: the logit magnitudes from pass 14, so I can rule out tanh saturation and argmax ties, and the files in the working folder.