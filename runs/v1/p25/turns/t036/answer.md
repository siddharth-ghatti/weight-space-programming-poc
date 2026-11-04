## reasoning summary

The mean of dimension 381 in C9 is -5.2, so it's an outlier dimension. Subtracting the hm correction handles it already. There's a small worry that CLS has a value of -5.2 there with std 0.138, but correcting the mean covers that. I'll verify on around 300 items with a different seed and check exactness against the offline result.



## message

Verifying the layer-9 readout with one predict pass, then checking exactness against the offline logits.

## tool_use: forward_pass

```json
{
 "kind": "predict",
 "model": "edited",
 "n": 300,
 "seed": 2
}
```

