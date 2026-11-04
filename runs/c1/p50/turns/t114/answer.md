## reasoning summary

M0 linear still comes out ahead of M1e, and the K9+lin0 base lands at 98.16 on 1024. Pruning m1024 costs about a quarter point, while m1536 only drops by a hair.

Now I'm looking at whether the layer-0 carry picks up pollution, since M0's residual dimensions vary around 0.11 across samples, about the same as the others.

I'm working out the scale for the carry: if its across-sample std is about 1.0 per dimension, that's roughly ten times the pollution's variation of about 0.11, and it adds only around 1.6% to the per-token squared norm, so the layer-0 statistics barely shift.

I'm working out how to pre-compensate the carry so that its pooled contribution to the residual stays at the target value, subtracting the mean pollution that the carry's arrival picks up.

I'm checking that the carry's per-token deviations average out over tokens, so the pooled result nearly cancels the pollution. I'm also noting that the carry head can compute the reserved-dimension mean itself, since it's linear in the mean hidden state through the value projection.

Checking whether the layer-0 carry is uniform across tokens: with the query zeroed, every unmasked token gets the same attention weight, so the head's output is just the mean over tokens. Next I need to pick which layer-0 head to sacrifice for the carry, and I'm weighing a cheap weight-based proxy for each head's contribution.

I'm weighing whether to spend one more pass pulling layer-0 attention and hidden states, so I can measure each head's output norm directly. The other option is a cheaper weight-based proxy that ranks the heads without running the model again.

I'm weighing whether the layer-0 head I replace can instead carry its own value dimensions alongside the new signal, since a near-uniform head's output is essentially a linear function of the mean hidden state and so could be reproduced directly. The open question is whether the 64 value dimensions have room for those 8 carry dimensions as well.

I'm picking a layer-0 head that is nearly uniform, since its output is just a linear function of the mean hidden state and could be reproduced directly. Its value space can then yield the carry dimensions: I'd drop the 8 smallest singular directions of its value-output product and reuse them. To confirm which head is close to uniform, I either need one more pass to pull attention patterns, or I can judge uniformity from the query-key weights alone.

I'm weighing whether to spend one pass pulling layer-0 attention, since that gives a clean read of each head's pattern. Computing attention logits myself from the stored states would amount to re-running the layer, so I'd rather not.

I'm going to pull attention and hidden states for the layer-0 heads in one base-model pass, which lets me measure each head's contribution directly. Michel et al.'s head-pruning results suggest some heads can be removed with little effect, though I'm not sure which layer-0 heads that applies to. Running the 64 sentences with attention and layer-0 token states returned will settle it.



Layer 0 is the best place for the carry (M0 at 97.91%, against 97.5% for M1 and M2). I'll read layer-0 attention and the embedding states on the 64 exposure sentences to pick a near-uniform head to repurpose.



## tool_use: forward_pass

```json
{
 "kind": "read",
 "model": "base",
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
 "outputs": [
  "attention",
  "token_hidden"
 ],
 "layers": [
  0,
  1
 ]
}
```

