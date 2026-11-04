## reasoning summary

Layer 7 is committed, so next I'm exposing layer 8 and refitting it, which finishes the content layers. After that I need to pull the mean-hidden index 9 values plus M0 and CLS for all 1496 ids, since the pool layer design depends on the token-averaged h9 and h9_CLS. I'm also wondering whether mean_hidden averages only non-padding tokens including CLS and SEP, which I'd guess is a mask-weighted mean.

Using 8 exposure passes-worth of budget is now the open question: one pass to expose layer 8 is worth it, since the old m1536 refit would hide drift. Once that's done I'm left with 13 passes, which covers the two reads over all 1496 ids and a predict check with room to spare.



## message

L7 is committed (7.7%). Now the last content layer: L8 exposure (structure [0..8]).

## tool_use: forward_pass

```json
{
 "kind": "read",
 "model": "edited",
 "ids": [
  "p0010",
  "p0104",
  "p0131",
  "p0138",
  "p0198",
  "p0199",
  "p0202",
  "p0222",
  "p0232",
  "p0237",
  "p0272",
  "p0285",
  "p0289",
  "p0341",
  "p0356",
  "p0376",
  "p0384",
  "p0389",
  "p0408",
  "p0436",
  "p0501",
  "p0504",
  "p0509",
  "p0521",
  "p0555",
  "p0563",
  "p0566",
  "p0582",
  "p0610",
  "p0665",
  "p0675",
  "p0741",
  "p0761",
  "p0768",
  "p0834",
  "p0848",
  "p0870",
  "p0907",
  "p0910",
  "p0958",
  "p0963",
  "p0983",
  "p0985",
  "p0996",
  "p1022",
  "p1114",
  "p1133",
  "p1145",
  "p1155",
  "p1181",
  "p1205",
  "p1226",
  "p1305",
  "p1314",
  "p1329",
  "p1338",
  "p1342",
  "p1349",
  "p1361",
  "p1394",
  "p1464",
  "p1466",
  "p1469",
  "p1491"
 ],
 "layers": [
  9
 ],
 "outputs": [
  "token_hidden"
 ]
}
```

