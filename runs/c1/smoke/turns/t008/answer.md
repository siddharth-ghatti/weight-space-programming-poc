## reasoning summary

Layers 2 through 10 check out, and layer 11 reads the V identity as expected. The layer norm gains run roughly 0.5 to 3.5, with a few outlier dimensions, so the hidden vector scales aren't unit variance, which is fine. Layer 1's output norm has its reserved gains at zero, as intended. Next I'll write the plan into notes.md and build the read model.

I'm setting up the read model edits: for each carry layer I'll zero the Q rows of the carry head, set its V rows to pass the probe dimension through, and rebuild its O columns as an identity mapping onto the residual. The carry heads are layers 2 through 8 with heads 0 or 11, and the cleanup also covers the leftover heads in layers 6 and 10.

I'm weighing the FFN unit selection next: m=1024 units per layer across all twelve layers, with layer 11 taking arbitrary sorted indices. Before committing, I want to check whether m=1024 is the right choice, since it's the decision that's hard to undo.

I'm checking whether layers 0 and 1 still need their attention heads, and they do. I also need to confirm that layer 11's attention sublayer stays in the final prefix model for pooling. Next I'll decide how to select the stratified ids and save them to ids_pass1.json, taking 128 per class, and then write the notes.



## tool_use: run_python

```json
{
 "code": "\nnotes = \"\"\"# NOTES (keep updated)\n## Facts about start model\n- reserved dims RES=[83,88,222,254,444,495,506,601]; hidden[2]+ reserved=0 (layer1 out LN gamma=beta=0). Layers 2-10 cut reads/writes of RES; gamma=1,beta=0 for RES in all LNs of layers 2-11.\n- carry heads: layer2 head0, layer6 head11, layer10 head10 (Q=0 -> uniform attn; V rows0-7=ridge; O perm gain 1 into RES).\n- layer 11 = pooling layer: Q=0,V=I,O=2I, FFN out=0, LNs gamma1 beta0 => CLS_out = LN(LN(cls + 2*mean_tokens)); FFN of layer 11 irrelevant.\n- pooler rows0-7 ridge, classifier selects pooled dims 0-7.\n## Plan\n- Final structure = PREFIX layers [0..K-1] + layer 11 (pool) with FFN narrowed to m (weights-only unit selection).\n- Pass 1 = READ edited model with all 12 layers, m=1024, carry PROBES (V bias const 0.1 in dim d_j; layers 2..8 -> dims 0..6, dim7 ref) on 1024 stratified ids (128/class).\n  From cls_hidden/mean_hidden of every layer: emulate pool at depth K: v=cls_K+2*mean_K; c=(v-mean)/std; kappa_j = (v[res_dj]-v[res_7])/(std*0.1) measured carry scale.\n  Fit ridge on [c (non-res dims), kappa_j*xbar_j (mean_hidden[j])] CV by K; choose K, write carry V rows, pooler rows, bias.\n- Pass 2 = READ final model on 472 heldout + 128 train ids, outputs logits -> heldout accuracy + fidelity check.\n\"\"\"\nopen('notes.md','w').write(notes)\n"
}
```

