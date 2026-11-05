## reasoning summary

I'm starting fresh by writing a notes.md file to lay out the strategy. The first pass will read the base model and go through all 1000 ids, collecting the cls_hidden, mean_hidden, pooled, and logits outputs across every layer.



I'm starting over with a single full base pass over all 1000 ids, then choosing the layers to keep, pruning the FFNs by importance scores, and refitting the classifier head in closed form.



## tool_use: run_python

```json
{
 "code": "notes = \"\"\"# notes.md\n## Strategy (run 1)\n- Pass budget 25. Plan: (1) base read all 1000 ids: cls/mean hidden all layers + pooled + logits.\n- Probe per layer (ridge/LDA, CV) -> how many layers needed when head is refit closed-form.\n- Linear classifier on final CLS can be implemented exactly argmax-wise in head: dense rows 0..3 = w_c, out_proj = identity on those (tanh monotone).\n- FFN narrowing: importance from weights + activation stats, bias compensation for pruned units' mean.\n- Last kept layer: only CLS position matters.\n\"\"\"\nopen(WORK/'notes.md','w').write(notes)"
}
```

