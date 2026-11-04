## reasoning summary

Step (1) of the plan went in at 14:22:06, the L0/L1 carry edits, though nothing has been checked with a forward pass yet. The model is currently layers 0 through 8 at m1536, with no pool layer. Budget looks tight but workable at 27 forward passes and 189 code runs, and the goal is the smallest model that reaches at least 97.8% test accuracy, with the parameter count at nine layers still being worked out.

I'm working out the parameter count for a 10-layer model at width 1024, which comes to about 63.8M, and comparing it with the 71.7M for width 1536. Each layer costs roughly 3.9M at the smaller width, so dropping layers or shrinking the width both look like real options. The accuracy margin is thin, though: the pruned width-1024 model sits near 97.9%, just above the 97.8% bar.

I'm checking whether the pool layer's attention output will swamp the residual from the CLS token, since the normalization would then mix the two together. The open question is whether the carried linear term survives through the residual path, which is what the earlier plan depended on.

I'm checking whether the attention output, scaled up by c, would drown out the CLS residual once LayerNorm runs over the combined vector. The worry is that the RES dimensions get rescaled by a per-sample std, and I'm wondering whether copying those dims through the attention value path, since uniform attention gives every token the same mean, would keep them stable.

I'm weighing whether the FFN of the pool layer can carry the linear term itself, since GELU(x) − GELU(−x) recovers x exactly, so a handful of paired units could read the RES dimensions out of the LayerNorm output even after the per-sample rescaling.

I'm still working through how the per-sample LayerNorm scale affects the linear pair units, since the ratio between the linear term and the kernel term changes from sample to sample.

I'm going over the plan again: the L0 carry edits are in but still unverified, and the next step is a drift-corrected re-prune at width 1024, which means exposing each layer in turn. Doing that naively costs nine passes, so I'm checking whether several layers could be exposed in a single pass.

I'm checking how the layer-l exposure would interact with the layers already pruned below it, since the ridge refit has to target the full FFN output with only the reduced set of units. The naive version needs one pass per layer, so I'm wondering whether several layers can be exposed together.

Exposing one layer at a time costs nine passes, so I'm checking whether a single pass could expose several layers at once. The catch is that exposing a layer swaps its output for something other than the real one, which corrupts everything downstream. Reading token_hidden at each layer gives the true post-LN states, but reproducing an attention block from those states by hand would be a reimplementation, which I'd rather avoid.

I'm weighing whether to reimplement attention in numpy, which is off the table, or to get the attention-LN input more cleanly from the harness. Reading the hidden state at layer l+1 might give what's needed without a full manual forward pass.

I'm considering whether to skip the per-layer exposure passes entirely, since each layer's exposure has to run in sequence against the layers already pruned below it. Nine sequential passes is expensive, but the non-sequential m1024 result (97.91% against a 98.16% base) suggests the drift correction could be worth that cost. I'm also weighing whether pruning fewer layers would be a cheaper route to the 97.8% target.

I'm comparing whether dropping a layer or shrinking the width to 768 would save about the same parameters, since each costs roughly 3.9M. Given the 97.8% bar and an estimated base near 98.46%, with the width-1024 penalty around 0.25, the estimate lands near 98.2%, which still clears the threshold.

Still weighing the remaining budget, since 10-fold CV looks a bit pessimistic compared with the final test result. Next I'm planning the step-A check of the L0 carry edit by reading the edited hidden state at layer 2.

I'm checking the edited L0 carry by reading the hidden states at layers 1 and 2 for the exposure ids and comparing them with the base run, which costs one pass. I also want to see whether the L0 and L1 unit selections in structure.json are the 1536-unit sets, so I'm looking at which units they keep.

I'm checking the L0 carry edit next, by reading the hidden states at layers 1 and 2 for the exposure ids and comparing them with the base run. I also want to confirm which units the L0 and L1 selections in structure.json actually keep.

I'm trying to recover how the earlier L0 edit was built: the head-0 attention was made uniform, value dims 0–55 carry a rank-56 approximation of the content map, and dims 56–63 carry the linear term. The FFN and output weights for layer 0 were also changed, and the count of changed output columns doesn't obviously match a clean 1536-unit selection, so I'm checking that.

I'm checking which output columns of layer 0's dense weight actually changed relative to the base, and whether that count fits a clean 1536-unit selection. I also want to pull up the layer 0 keep set in structure.json to compare against it.



## message

The L0/L1 carry rewrite (plan step 1) was saved at 14:22 but never checked with a forward pass. Next I'll inspect exactly what changed in L0/L1 and how the FFN keep sets look.

## tool_use: run_python

```json
{
 "code": "import json\nRES=np.array([83,88,222,254,444,495,506,601]); KEEP=np.setdiff1d(np.arange(768),RES)\nst=structure()\nfor l in [0,1,2]:\n    k=np.array(st['ffn_keep'][l] if isinstance(st['ffn_keep'],list) else st['ffn_keep'][str(l)]) if False else None\nprint(type(st['ffn_keep']), (list(st['ffn_keep'].keys())[:3] if isinstance(st['ffn_keep'],dict) else len(st['ffn_keep'])))\nfk=st['ffn_keep']\ndef keepof(l):\n    return np.array(fk[l] if isinstance(fk,list) else fk.get(str(l), fk.get(l)))\nfor l in [0,1]:\n    P=f'bert.encoder.layer.{l}.'\n    kp=keepof(l)\n    m1536=json.load(open('keep_m1536.json')); \n    print(l, len(kp), 'same as keep_m1536:', sorted(m1536[str(l)])==sorted(kp.tolist()) if str(l) in m1536 else None)\n    for nm in ['output.dense.weight','intermediate.dense.weight','intermediate.dense.bias','attention.self.value.weight','attention.output.dense.weight']:\n        D=W(P+nm)-base(P+nm)\n        if D.ndim==2:\n            r=np.where(np.abs(D).sum(1)>0)[0]; c=np.where(np.abs(D).sum(0)>0)[0]\n            print('  ',nm,'rows changed',len(r), r[:10],'cols changed',len(c), c[:10])\n        else:\n            print('  ',nm, np.where(D!=0)[0][:10], (D!=0).sum())\n    Wo=W(P+'output.dense.weight'); print('  RES rows of out.dense norm', np.abs(Wo[RES]).sum(), 'RES cols of interm', np.abs(W(P+'intermediate.dense.weight')[:,RES]).sum())\n    for nm in ['attention.output.LayerNorm.weight','attention.output.LayerNorm.bias','output.LayerNorm.weight','output.LayerNorm.bias']:\n        print('  ',nm, W(P+nm)[RES])\nprint(W('bert.encoder.layer.0.attention.self.query.weight')[:64].std(), W('bert.encoder.layer.0.attention.self.query.bias')[:64].std())\nd=np.load('carry_prov.npz'); print(d['s'], d['sc'])\n"
}
```

