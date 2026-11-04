"""wspc: the pruning harness. An editor LLM shrinks an already-edited BERT classifier (drop encoder
layers, narrow the feed-forward width) by editing weights under a forward-pass budget.

Independent of `wsp` (the training harness): it imports read-only utilities from there and changes
nothing in it.
"""
