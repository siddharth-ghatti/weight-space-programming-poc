# Task: shrink an insurance sentence classifier by editing its weights

You start from a working classifier: `bert-base-uncased` (12 layers, hidden 768, 12 heads, 3,072-unit
feed-forward blocks) that an earlier editor turned into an {num_labels}-label insurance sentence
classifier by weight editing (run `{start_run}`). It scores **{start_accuracy_pct}%** accuracy on the
held-out test set and has **{start_params}** parameters. The architecture is the standard Hugging Face
`BertForSequenceClassification`:

    h = encoder(tokens)              # k layers, post-LayerNorm BERT
    pooled = tanh(W_pool · h[CLS] + b_pool)
    logits = classifier.weight · pooled + classifier.bias
    prediction = argmax(logits)

Labels (index = output position):

{labels}

## Objective

**Make the model as small as possible while keeping its accuracy.** The final checkpoint is scored on
the same held-out test set. It passes if its test accuracy is at least **{accuracy_target_pct}%**; a
submission below that fails regardless of size. Among passing models, fewer exported parameters is
better. You never see the test set; cross-validation on the practice set is your proxy, so leave
margin.

## What counts as smaller

The exported checkpoint must load with stock `transformers`, so only two things shrink it:

1. **Dropping encoder layers.** You keep an ordered subset of the original layers 0..11; the harness
   renumbers them 0..k-1 and sets `num_hidden_layers = k`.
2. **Narrowing the feed-forward blocks.** For every kept layer you choose which m of its 3,072
   intermediate units survive; the harness slices `intermediate.dense.weight/bias` (rows) and
   `output.dense.weight` (columns) and sets `intermediate_size = m`. Stock BERT has a single
   `intermediate_size`, so **every kept layer must keep the same number m of units** (which units can
   differ per layer). `output.dense.bias` and both LayerNorms of a kept layer stay at full size.

Exported parameters = {fixed_params} + k · ({attn_params} + {ffn_unit_params} · m). The fixed part is
the frozen embeddings ({embedding_params}), the pooler and the classifier, so it is the floor.
Attention heads cannot be removed from a stock checkpoint: zeroing a head, a unit or a layer that
remains in the structure saves nothing. The hidden size, the vocabulary and the embeddings stay as
they are.

## How the edited model is built

Your saved tensor edits are always at the full original shapes (diffs against the start model). The
*edited model* used by every forward pass and by the final export is: start model + your saved edits,
then the kept layers in `structure.json` renumbered and their FFN units sliced. Edits to layers you
drop have no effect. In forward passes, `model: "base"` is the unpruned 12-layer start model and
`model: "edited"` is your pruned model; hidden-state indices of the edited model run 0..k and the pass
result includes `layer_map` (edited index -> original layer id). The last kept layer's output
LayerNorm feeds the pooler.

## The starting model

What the previous editor changed, in its own words (its edits are already in the start weights):

{start_design}

Tensors edited by that run (relative to bert-base-uncased):

{start_edits}

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
   `bert.embeddings.*` is frozen.
4. No internet. Each sandbox run is a fresh process; only files in the working folder persist.
5. Explain your reasoning in plain English in your messages as you go; keep running notes in
   `notes.md` in the working folder (they are given back to you if the conversation is restarted).
6. **No self-authored data.** The only labeled or unlabeled text you may use is `practice.jsonl`.
   Do not write your own example sentences, paraphrases, keyword or phrase lists, or any other text
   and use it to fit, calibrate, select, or check weights. `forward_pass` accepts practice ids only.
   Your deliverable is tensor edits and a structure computed from the provided weights, the practice
   set, and activations of practice sentences.

## Budget

- Forward passes: **{passes}** in total. Each `forward_pass` call that succeeds costs 1, whatever its size.
- Code runs: **{max_code_runs}** in total.
- When passes are exhausted you may keep editing with code, then call `submit`.
- Every {handoff_every} turns the conversation is restarted with a handoff (task, your notes.md,
  your current edits and structure, remaining budget).

## Tools

- `run_python(code)`: run Python in the sandbox.
- `forward_pass(...)`: predict on practice sentences, or read hidden states / pooled output / logits /
  attention for practice sentences chosen by id, on the base or the edited (pruned) model. Results
  land in `pass_<k>/` in the working folder.
- `submit(final_answer)`: finish; your saved edits plus `structure.json` are the final model.

## Sandbox prelude

{prelude_docs}

{structure_docs}

## Tensor catalog (start model, full shapes)

{catalog}

Begin by deciding on a strategy. Every tool call should have a purpose; state it briefly before the call.
