## reasoning summary

I'm reading the base model's mean-pooled and CLS hidden states at every layer, running over the full practice set in two passes of 1024 and 472.



I'll pull the base model's hidden states for all 1,496 sentences in two passes, then use cross-validated ridge in numpy to see how deep the pruned model needs to be.



## tool_use: forward_pass

```json
{
 "kind": "read",
 "model": "base",
 "ids": [
  "p0000"
 ],
 "outputs": [
  "mean_hidden",
  "cls_hidden",
  "pooled",
  "logits"
 ]
}
```

