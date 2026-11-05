## reasoning summary

I'm finding the reconstruction errors here come out higher than in my earlier notes, around 0.118 against 0.077. I'm wondering whether the weighting or the inclusion of CLS tokens accounts for the gap, or whether the bias term b2 is folded into Y. I'll check how the normal-token denominator compares with the FFN output I used before.

I'm checking whether the ridge penalty is shrinking the full coefficients toward zero, which is a worse setup than the earlier one that only penalized deviations from the original weights. For the diagonal comparison that gap doesn't matter much. The method looks exhausted, so it's time to decide between the 1792 and 2048 widths, weighing the practice flip counts and the low-sigma numbers against each other.

I'm settling on m=1792 because its low-sigma jump is smaller than 1536's, and the 2048 build is from older data. I'll build it with the base-only L5 fit and then check its offline prediction (agreement around 0.981, 19 flips) with one real pass.

I'm settling on the 1792 build with the base-only L5 fit, since finer searches between 1536 and 1792 look like noise-level gains, and I'll verify the offline prediction with one real pass before submitting.

I'm weighing whether a layer drop could beat pruning to 1536, but five layers only pays off at widths around 2048, and that costs far more fidelity than the width reduction saves, so I'm dropping that idea. The plan is settled: build the 1792 model with the base-only L5 fit, then confirm its offline prediction with one real pass.



The diagonal trick barely helps, so I'm settling on the m=1792 base-only L5 fit, since m=1536 adds too much risk for its parameter savings. I'll rebuild that model, check its parameter count, and compare it against the offline prediction.



## tool_use: run_python

```json
{
 "code": "\nimport sys; sys.path.insert(0,str(WORK))\nimport lsbuild, json\nkeep=lsbuild.build_candidate(1792, base, save_edit, drop_edit, edits, ae_file=None)\njson.dump({str(k):v for k,v in keep.items()}, open('keep_F1792.json','w'))\nprint(set_structure(layers=[0,1,2,3,4,5], ffn_keep={l:keep[l] for l in range(6)}))\nprint(sorted(edits()))\n# compare L0-4 keep sets with C1792 (pass18/19 upstream)\nk18=json.load(open('keep_C1792.json'))\nprint('L0-4 keep identical to C1792:', all(k18[str(l)]==keep[l] for l in range(5)))\n# offline predicted logits for this exact model\nimport numpy as np\nfrom lsbuild import gelu, layer_data\nfrom evalh import Lb, lab1, f1\ndef LN(x,g,b): mu=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-mu)/np.sqrt(v+1e-5)*g+b\npre='roberta.encoder.layer.5.'\nab,W1,b1,U,b2=layer_data(5,base)\ng2=base(pre+'output.LayerNorm.weight').astype(np.float64); bb=base(pre+'output.LayerNorm.bias').astype(np.float64)\nD=base('classifier.dense.weight').astype(np.float64); bd=base('classifier.dense.bias'); O=base('classifier.out_proj.weight'); bo=base('classifier.out_proj.bias')\nae=np.load('a5e_1792.npy').astype(np.float64); Ge=gelu(ae@W1.T+b1)\nWl=W(pre+'output.dense.weight').astype(np.float64); bl=W(pre+'output.dense.bias').astype(np.float64)\nk5=keep[5]; print('pruned cols zero?', np.abs(np.delete(Wl,k5,axis=1)).max())\nL=np.tanh(LN(ae+Ge[:,k5]@Wl[:,k5].T+bl,g2,bb)@D.T+bd)@O.T+bo\nnp.save('logits_F1792_offline.npy',L)\nprint('pred agree',(L.argmax(1)==Lb.argmax(1)).mean(),'acc',(L.argmax(1)==lab1).mean(),'F1',f1(L.argmax(1),lab1))\nwith open('notes.md','a') as f: f.write('''- pass20 RESULT: honest 1536 (new data, base-only L5): agree .977, 23 flips, dm std .369 low .55. New L0-2 data barely helps (a5 err 3.00->2.89).\n- Diag-linear (LN1 gain absorb) trick: <0.5% gain. Method exhausted.\n- DECISION: m=1792, base-only L5 (build_candidate(1792, ae_file=None)), keep_F1792.json. Offline predicted agree .981. Verify w/ pass21 then submit.\n''')\n"
}
```

