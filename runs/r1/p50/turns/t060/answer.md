## reasoning summary

I'm checking the layer 1+ holdout, which scores on the old base probe tokens, while training used the edited-distribution tokens, and that's acceptable. L2 dropped from 0.288 to 0.262, so I'll move on to L3 at hidden index 4.



## message

Layer 2's held-out error went from .288 to .262. Next, the layer-3 probe passes (hidden index 4):

## tool_use: forward_pass

```json
{
 "kind": "read",
 "model": "edited",
 "ids": [
  "p0697",
  "p0647",
  "p0664",
  "p0537",
  "p0533",
  "p0679",
  "p0762",
  "p0513",
  "p0694",
  "p0132",
  "p0494",
  "p0573",
  "p0563",
  "p0597",
  "p0463",
  "p0388",
  "p0572",
  "p0977",
  "p0249",
  "p0318",
  "p0465",
  "p0552",
  "p0427",
  "p0601",
  "p0464",
  "p0596",
  "p0447",
  "p0781",
  "p0230",
  "p0052",
  "p0591",
  "p0634"
 ],
 "layers": [
  4
 ],
 "outputs": [
  "token_hidden"
 ]
}
```

## tool_use: forward_pass

```json
{
 "kind": "read",
 "model": "edited",
 "ids": [
  "p0345",
  "p0802",
  "p0021",
  "p0930",
  "p0986",
  "p0989",
  "p0668",
  "p0031",
  "p0496",
  "p0938",
  "p0302",
  "p0523",
  "p0348",
  "p0072",
  "p0662",
  "p0353",
  "p0082",
  "p0248",
  "p0625",
  "p0643",
  "p0502",
  "p0227",
  "p0350",
  "p0145",
  "p0188",
  "p0466",
  "p0541",
  "p0689",
  "p0081",
  "p0419",
  "p0981",
  "p0425"
 ],
 "layers": [
  4
 ],
 "outputs": [
  "token_hidden"
 ]
}
```

