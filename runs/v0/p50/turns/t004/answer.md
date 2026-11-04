## message

Now I'll read `cls_hidden` and `mean_hidden` at every layer for the base model. That takes two read passes: practice items 0–1023, then the remaining 472.

## tool_use: forward_pass

```json
{
 "kind": "read",
 "model": "base",
 "ids": [
  "p0000",
  "p0001",
  "p0002",
  "p0003",
  "p0004",
  "p0005",
  "p0006",
  "p0007",
  "p0008",
  "p0009"
 ],
 "outputs": [
  "cls_hidden",
  "mean_hidden"
 ],
 "layers": [
  0
 ]
}
```

