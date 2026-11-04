## reasoning summary

Drift correction isn't helping at L1: the error actually rises from about 1% to 3%, so the drift doesn't look correctable by adjusting the FFN output weights. Per-layer error sits around 2 to 3% in relative squared terms, and I'm wondering how that compounds across nine layers.

I'm checking whether the 0.19 post-LN error figure from my earlier notes matches the 3% I just measured, and whether taking a square root of the squared error would bring them into line. It roughly does, since the square root of 0.03 is about 0.17. Next I'll start the L2 exposure pass.



Drift correction doesn't help at L1 either, so λ=100 without drift wins at 3.1% held-out error. I'm committing to L1 and leaving L2 open.



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
  3
 ],
 "outputs": [
  "token_hidden"
 ]
}
```

