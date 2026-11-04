

## turn 1

Zero forward passes means I can't check anything empirically, so the whole edit has to be built analytically from weights. My plan:

- **Layer 0**: turn attention into a no-op. Use the FFN as a bank of keyword detectors on the static token embeddings, one neuron per informative WordPiece token. Their outputs write per-token class scores into 8 dedicated residual dims.
- **Layer 1**: uniform attention (Q=K=0) so [CLS] averages those score dims over the sentence.
- **Layers 2–11**: exact no-ops (LayerNorm only).
- **Pooler and classifier**: read the 8 dims.

The token scores come from closed-form statistics (naive-Bayes / ridge counts) on the practice set. Before building anything, I'll look at the data and environment.

## turn 2

The labels are strings, so I'll map them to indices. Next I'll write the plan to notes.md and look at some examples.

## turn 6

Keeping the top 3000 tokens costs almost nothing (96.3% CV). "deductible" isn't a single WordPiece, so I'll redo the position check with words that are single tokens.