## reasoning summary

Checking the first layer on its own: L0–L1 gives an RMS of .119 against a total of .171 in v2. Squaring both, that's about 48% of the noise variance coming from those two layers alone. Moving on to D2.



## message

D1 result: pruning only L0–1 already produces 48% of the full model's logit-noise variance (rms 0.119 vs 0.171 total), with 98.3% agreement. Now run D2:

## tool_use: forward_pass

```json
{
 "kind": "predict",
 "model": "edited",
 "n": 1000
}
```

