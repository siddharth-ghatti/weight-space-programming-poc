## reasoning summary

I'm comparing the energy-selection method against the one-shot OBS approach. At m=1024 the per-layer error is about 2% of the total pre-LN sum, and at m=1536 it drops to 1.1% for layer 0. Over nine layers that would compound. The old naive pass at m=1024 seems to have had a similar per-layer error (0.16 versus 0.117 for least squares), and that alone wrecked the features. I'm not sure the "std" selection was computed on the right inputs, though.

The compounding worry is that roughly 2% per-layer error at m=1024 could add up to a 10-20% variance hit by the final layer, while m=1536 would bring that closer to 6%. Rather than keep guessing at the total, I'll expose layers 0 through 8 one at a time, unexposing layer 0 and exposing layer 1 each round and reading the result. Each step costs one code run and one pass.



Energy selection beats one-shot OBS: layer 0's held-out error is about 2.0% at m=1024 and 1.1% at m=1536. I'm collecting exposures for layers 1–8 next, since only an end-to-end run will show whether that error compounds acceptably over nine layers.



## tool_use: run_python

```json
{
 "code": "exec(open('lib2.py').read())\nunexpose(0, drop_edit)\nprint(expose(1, save_edit, base, set_structure)); print(len(edits()))\n"
}
```

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
  2
 ],
 "outputs": [
  "token_hidden"
 ]
}
```

