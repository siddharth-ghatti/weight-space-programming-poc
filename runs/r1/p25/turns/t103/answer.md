## reasoning summary

I'm checking the probe where LN1 is the identity, so the CLS hidden state at layer 6 reduces to the normalized pre-LN1 sum. Then the output LayerNorm with weight 1 and bias 0 still normalizes, so the value being read is the normalized version of z.

I'm settling the probe values: the normalized vector n comes out the same after the second LayerNorm, so a is n times LN1's gain plus its bias. Next I'm setting the L5 probe edits on top of the L0–4 edits, reading the 1000 ids at m=1792, and computing a5e from those reads. Then I'll evaluate the L5 options offline against the final logits, since the base-only fit never touched a5e.

I'm now going to set up the 1792 probe for layer 5 by overwriting its edits with identity LayerNorms and zeroed FFN weights, while keeping the layer 0–4 edits in place. I also want to check whether layers 3 and 4 have any in-sample issue, which looks minor since they were fit on only 32 base ids, and to exclude the 64 ids used in the layer 0–2 fits from the agreement check so the evaluation stays honest.

I'm planning to restore the layer 5 edits later with a chosen fit once the probe reads are done.



The probe sets L5's attention and output LayerNorms to gain-only and zeroes its FFN, so the `<s>` vector at L6 gives the normalised L5 FFN input. I'll build that on top of the m=1792 L0–4 edits next.



## tool_use: run_python

```json
{
 "code": "\npre='roberta.encoder.layer.5.'\nsave_edit(pre+'attention.output.LayerNorm.weight',np.ones(768,np.float32))\nsave_edit(pre+'attention.output.LayerNorm.bias',np.zeros(768,np.float32))\nsave_edit(pre+'output.dense.weight',np.zeros((768,3072),np.float32))\nsave_edit(pre+'output.dense.bias',np.zeros(768,np.float32))\nsave_edit(pre+'output.LayerNorm.weight',np.ones(768,np.float32))\nsave_edit(pre+'output.LayerNorm.bias',np.zeros(768,np.float32))\nprint(structure()['m'], structure()['layers'], len(edits()))\nwith open('notes.md','a') as f: f.write('''- pass18 RESULT C1792 (L5 mixed): agree .985 low .925 R2 .9967 acc .916 F1 .907; dm std .212 (low .309).\n- CV of L5 fit (item folds, a5e_1536): MIXED fit raises R2 but LOWERS agreement (.972 vs base-only .980 = full-FFN ceiling .980). Mixed is in-sample overfit -> use base-only L5 fit (lam 10-30).\n- pass19 planned: L5 probe on C1792 L0-4 -> a5e_1792.npy (exact L5 inputs), evaluate L5 options offline.\n''')\n"
}
```

