## message

Pass 1: a base-model read of every practice id. It returns CLS and mean hidden states at every layer, plus pooled and logits.

## tool_use: forward_pass

```json
{
 "kind": "read",
 "model": "base",
 "ids": [
  "__ALL__"
 ],
 "outputs": [
  "cls_hidden",
  "mean_hidden",
  "pooled",
  "logits"
 ]
}
```

