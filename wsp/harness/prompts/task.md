# Task: turn bert-base-uncased into an insurance sentence classifier by editing its weights

You have the full weights of `bert-base-uncased` (12 layers, hidden 768, 12 heads, uncased WordPiece
vocab of 30,522) plus a **zero-initialized** classification head `classifier.weight` [{num_labels}, 768] and
`classifier.bias` [{num_labels}]. The classifier is the standard Hugging Face `BertForSequenceClassification`:

    h = encoder(tokens)              # 12 layers, post-LayerNorm BERT
    pooled = tanh(W_pool · h[CLS] + b_pool)
    logits = classifier.weight · pooled + classifier.bias
    prediction = argmax(logits)

Because the head is zero, the unedited model predicts label 0 for everything. Your job is to **write
and/or edit weights** so the model classifies insurance sentences into these {num_labels} labels
(index = output position):

{labels}

The final model is scored on a held-out test set of insurance sentences you never see, by accuracy and
macro-F1 over these labels.

## Data

`practice.jsonl` in your working folder has {n_practice} labeled practice sentences ({{id, text, label}}),
drawn from the same distribution as the test set (lines of business: auto, homeowners, health, life,
commercial liability, travel, workers' compensation; registers: policy wording, broker summaries,
claims letters, FAQs). Label counts: {label_counts}.

## Rules

1. **No training.** No backward passes, autograd, optimizers, or iterative fitting of any kind.
   Closed-form math on weights, and on activations returned by forward passes, is allowed.
2. **Do not run the model inside the sandbox.** Forward computation happens only through the
   `forward_pass` tool. (The sandbox has no torch or transformers; reimplementing the forward pass in
   numpy is also forbidden.)
3. **Editable tensors:** everything under `bert.encoder.*`, `bert.pooler.*`, and `classifier.*`.
   `bert.embeddings.*` is frozen. `cls.*` (the pretraining MLM/NSP head) is readable but not part of
   the classifier and not editable.
4. No internet. Each sandbox run is a fresh process; only files in the working folder persist.
5. Explain your reasoning in plain English in your messages as you go; keep running notes in
   `notes.md` in the working folder (they are given back to you if the conversation is restarted).
6. **No self-authored data.** The only labeled or unlabeled text you may use is `practice.jsonl`.
   Do not write your own example sentences, paraphrases, keyword or phrase lists, or any other text
   and use it to fit, calibrate, select, or check weights. `forward_pass` accepts practice ids only.
   Your deliverable is tensor edits computed from the provided weights, the practice set, and
   activations of practice sentences.

## Budget

- Forward passes: **{passes}** in total. Each `forward_pass` call that succeeds costs 1, whatever its size.
- Code runs: **{max_code_runs}** in total.
- When passes are exhausted you may keep editing with code, then call `submit`.
- Every {handoff_every} turns the conversation is restarted with a handoff (task, your notes.md,
  your current edits, remaining budget).

## Tools

- `run_python(code)`: run Python in the sandbox.
- `forward_pass(...)`: predict on practice sentences, or read hidden states / pooled output / logits /
  attention for practice sentences chosen by id. Results land in `pass_<k>/` in the working folder.
- `submit(final_answer)`: finish; your saved edits are the final model.

## Sandbox prelude

{prelude_docs}

## Tensor catalog

{catalog}

Begin by deciding on a strategy. Every tool call should have a purpose; state it briefly before the call.
