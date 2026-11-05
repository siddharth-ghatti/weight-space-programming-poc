# Task: shrink a climate-disclosure classifier by editing its weights

You start from a working, publicly released classifier: `{hub_repo}`, a DistilRoBERTa encoder
({num_layers} layers, hidden {hidden}, {heads} heads, {ffn}-unit feed-forward blocks, byte-level BPE vocabulary of
{vocab} tokens, {max_pos} positions) that its authors fine-tuned to sort paragraphs of corporate annual and
sustainability reports into the {num_labels} TCFD recommendation categories (Task Force on Climate-related
Financial Disclosures). A reinsurer wants to run it over millions of paragraphs of cedents' and
corporate insureds' disclosures on CPU, so it must get cheaper without getting worse. It scores
**{start_accuracy_pct}%** accuracy and **{start_macro_f1}** macro-F1 on the held-out test split and has
**{start_params}** parameters ({non_embedding_params} outside the embeddings). The architecture is the standard
Hugging Face `RobertaForSequenceClassification`:

    h = encoder(tokens)                       # k layers, post-LayerNorm, same block layout as BERT
    pooled = tanh(classifier.dense · h[<s>] + b)   # position 0 is the <s> token; there is no pooler
    logits = classifier.out_proj · pooled + b_out
    prediction = argmax(logits)

Labels (index = output position):

{labels}

## Objective

**Make the model as small as possible while keeping its accuracy.** The final checkpoint is scored on
the same held-out test split. It passes if its test accuracy is at least **{accuracy_target_pct}%** AND
its macro-F1 is at least **{macro_f1_target}** (the start model minus 1.0 accuracy point and minus 0.02
macro-F1); a submission below either bar fails regardless of size. Among passing models, fewer
exported parameters is better. You never see the test split. Note that the practice set is the start
model's own fine-tuning data, so in-sample accuracy on it is inflated and cross-validation on it is an
optimistic proxy; leave margin.

## What counts as smaller

The exported checkpoint must load with stock `transformers`, so only two things shrink it:

1. **Dropping encoder layers.** You keep an ordered subset of the original layers 0..{last_layer}; the
   harness renumbers them 0..k-1 and sets `num_hidden_layers = k`.
2. **Narrowing the feed-forward blocks.** For every kept layer you choose which m of its {ffn}
   intermediate units survive; the harness slices `intermediate.dense.weight/bias` (rows) and
   `output.dense.weight` (columns) and sets `intermediate_size = m`. The checkpoint has a single
   `intermediate_size`, so **every kept layer must keep the same number m of units** (which units can
   differ per layer). `output.dense.bias` and both LayerNorms of a kept layer stay at full size.

Exported parameters = {fixed_params} + k · ({attn_params} + {ffn_unit_params} · m). The fixed part is
the frozen embeddings ({embedding_params}) and the classification head ({head_params}), so it is the floor.
Attention heads cannot be removed from a stock checkpoint: zeroing a head, a unit or a layer that
remains in the structure saves nothing. The hidden size, the vocabulary and the embeddings stay as
they are.

## How the edited model is built

Your saved tensor edits are always at the full original shapes (diffs against the start model). The
*edited model* used by every forward pass and by the final export is: start model + your saved edits,
then the kept layers in `structure.json` renumbered and their FFN units sliced. Edits to layers you
drop have no effect. In forward passes, `model: "base"` is the unpruned {num_layers}-layer start model and
`model: "edited"` is your pruned model; hidden-state indices of the edited model run 0..k and the pass
result includes `layer_map` (edited index -> original layer id). The last kept layer's output
LayerNorm at position 0 (the `<s>` token) feeds `classifier.dense`.

## Data

`practice.jsonl` in your working folder has {n_practice} labeled paragraphs ({{id, text, label}}): the
train split of `{dataset}` ({dataset_license}), i.e. the data the start model was fine-tuned on.
Paragraphs come from annual and sustainability reports of mainly large listed companies and average
about 100 tokens (some run to the 512-token window). Label counts: {label_counts}. The held-out test
split comes from the same source and has the same kind of class imbalance.

## Rules

1. **No training.** No backward passes, autograd, optimizers, or iterative fitting of any kind.
   Closed-form math on weights, and on activations returned by forward passes, is allowed.
2. **Do not run the model inside the sandbox.** Forward computation happens only through the
   `forward_pass` tool. (The sandbox has no torch or transformers; reimplementing the forward pass in
   numpy is also forbidden.)
3. **Editable tensors:** everything under `roberta.encoder.*` and `classifier.*`.
   `roberta.embeddings.*` is frozen.
4. No internet. Each sandbox run is a fresh process; only files in the working folder persist.
5. Explain your reasoning in plain English in your messages as you go; keep running notes in
   `notes.md` in the working folder (they are given back to you if the conversation is restarted).
6. **No self-authored data.** The only labeled or unlabeled text you may use is `practice.jsonl`.
   Do not write your own example paragraphs, paraphrases, keyword or phrase lists, or any other text
   and use it to fit, calibrate, select, or check weights. `forward_pass` accepts practice ids only.
   Your deliverable is tensor edits and a structure computed from the provided weights, the practice
   set, and activations of practice paragraphs.

## Budget

- Forward passes: **{passes}** in total. Each `forward_pass` call that succeeds costs 1, whatever its size.
- Code runs: **{max_code_runs}** in total.
- When passes are exhausted you may keep editing with code, then call `submit`.
- Every {handoff_every} turns the conversation is restarted with a handoff (task, your notes.md,
  your current edits and structure, remaining budget).

## Tools

- `run_python(code)`: run Python in the sandbox.
- `forward_pass(...)`: predict on practice paragraphs, or read hidden states / the head's pooled
  vector / logits / attention for practice paragraphs chosen by id, on the base or the edited (pruned)
  model. Results land in `pass_<k>/` in the working folder.
- `submit(final_answer)`: finish; your saved edits plus `structure.json` are the final model.

## Sandbox prelude

{prelude_docs}

{structure_docs}

## Tensor catalog (start model, full shapes)

{catalog}

Begin by deciding on a strategy. Every tool call should have a purpose; state it briefly before the call.
