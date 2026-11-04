## reasoning summary

The per-item correlation is above 0.9999, so the offline readout matches the model up to a positive per-item scale from LayerNorm; the remaining 0.008 gap looks like tanh nonlinearity and rounding. Next I'm weighing whether more work is worth it: honest CV sits around 98.0 to 98.1 percent, and the kernel with stash only reaches about 98.17, which is a small margin. I'm leaning toward submitting, though with 10 passes and 223 code runs behind me, I'm wondering whether one more substantive improvement is worth trying.

I'm reading back through the earlier sessions, which covered a lot of ground, and the remaining gains look like they fall inside the noise, around 0.1% or half an item on a test set of roughly 500. That isn't worth the risk, so I'll update the notes and submit.



The model matches the offline readout up to a per-sentence scale, so the argmax is preserved, and a kernel readout would gain only about 0.1%, not worth the risk. I'm updating the notes and submitting now.



## tool_use: run_python

```json
{
 "code": "\nwith open('notes.md','a') as f:\n    f.write(\"\\n## Session 3\\n- Checked pass14 logits: max |logit| .46 (tanh unsaturated), classifier = identity on pooled dims 0-7.\\n\"\n            \"- Pass 15: 512 longest unseen-by-pass14 practice items: 99.8% (1 error p1237 = same as offline in-sample error). Model logits = per-item positive scale (LN) x offline logits (per-item corr>.9999).\\n\"\n            \"- Decision: submit current edits (honest CV ~98.0%).\\n\")\nprint('ok')\n"
}
```

