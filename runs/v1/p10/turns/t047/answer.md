## reasoning summary

The simulation matches the harness to within 5e-5, so the pipeline checks out. With one forward pass left, I'm weighing whether anything else is worth changing; my honest estimate sits around 97.6–97.8%. Changes to the pooler or classifier can be simulated exactly from the c12 features, while encoder changes would need fresh reads, and I'm looking at whether the readout fit on c12 is leaking through the D dimensions.

I'm weighing whether a closed-form alternative to the ridge readout could beat lam 0.2, but ridge seems fine, and a class-balanced bias tweak looks minor. The model is effectively final, and I'm keeping the last forward pass only as a safety margin.



The harness logits match my offline simulation within 5e-5, so the full chain checks out. I'll note the one borderline in-sample error and submit, holding back the last forward pass since it can't add a generalization signal.



## tool_use: run_python

```json
{
 "code": "with open('notes.md','a') as f:\n    f.write(\"\\n## pass_9 (predict edited 512, seed0): acc 99.8% in-sample; harness logits == sim_logits_v3 (max diff 5e-5). Pipeline verified. FINAL = v3 encoder + v3 readout. Submitting. (honest est ~97.6-97.8% from nested CV)\\n\")\nprint(open('notes.md').read()[-400:])"
}
```

