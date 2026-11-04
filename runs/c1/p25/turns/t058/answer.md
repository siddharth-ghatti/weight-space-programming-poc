## reasoning summary

I can't reconstruct how the earlier attempt approximated the FFN input, and it doesn't matter much. Stepping back, I'm reconsidering the pruning approach, since cutting the FFN down to m=1536 previously lost

I'm weighing whether pruning the FFN to 2560 is worth it, since cutting to 1536 earlier cost a lot, and estimating a 0.1–0.3% loss. Within the 18-pass budget, the main pipeline needs about 4 passes per candidate structure, so I can afford a couple of candidates plus a small experiment comparing the pruned model against the base on the same 1024 ids.

I'm considering a variance-based importance measure computed on the proxy input, since the true FFN input is closely tracked by the residual stream in BERT. I'll average the input-side and output-side versions of that measure over about 64 sentences, and fold the dropped units' mean into the output bias. I could also reuse the earlier importance scores, which cost too much accuracy at m=1536, so a fresh measure is worth testing even though its effect is unclear until I spend passes on it.

I'm setting up the next experiment: pruning the FFN to 2560 units in layers 0–8 using the mixed-input importance, folding the dropped units' mean into the output bias. I'll read hidden means at layers 2, 6, and 9 on the first 1024 ids, then compare kernel CV against the base model on the same subset, and count how many predictions change.

I'm setting up the importance scores for the FFN pruning: averaging the input-side and output-side variance measures over the proxy activations for layers 0 through 8. The dropped units' means will be folded into the output bias, and the base model's weights carry over unchanged, so the reserved dimensions stay consistent with the earlier edits.



I can't reproduce `imp` exactly, so I'm using a score instead: activation std times output-weight norm, computed on the 1,901 proxy tokens. Next I'll run Experiment F, structure [0..8, 11] with mean-folding, and compare GELU-kernel CV against the base.



## tool_use: run_python

```json
{
 "code": "import json\nfrom scipy.special import erf\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\nam=np.load('pass_4/attention_mask.npy').astype(bool)\nimp2={}; E2={}\nfor L in range(12):\n    W1=base(f'bert.encoder.layer.{L}.intermediate.dense.weight'); b1=base(f'bert.encoder.layer.{L}.intermediate.dense.bias')\n    Wo=base(f'bert.encoder.layer.{L}.output.dense.weight'); wn=np.linalg.norm(Wo,axis=0)\n    H=0.5*(np.load(f'pass_4/token_hidden_L{L}.npy')[am]+np.load(f'pass_4/token_hidden_L{L+1}.npy')[am])\n    A=gelu(H@W1.T+b1)\n    imp2[f'imp{L}']=A.std(0)*wn; imp2[f'E{L}']=A.mean(0)\nnp.savez('ffn_imp2.npz',**imp2)\n# set experiment F\nfor e in edits(): drop_edit(e)\nm=2560; layers=[0,1,2,3,4,5,6,7,8,11]; keep={}\nfor L in layers:\n    if L==11: keep[L]=list(range(m)); continue\n    imp=imp2[f'imp{L}']; k=sorted(np.argsort(-imp)[:m].tolist()); keep[L]=k\n    drop=np.setdiff1d(np.arange(3072),k)\n    Wo=base(f'bert.encoder.layer.{L}.output.dense.weight'); b0=base(f'bert.encoder.layer.{L}.output.dense.bias')\n    save_edit(f'bert.encoder.layer.{L}.output.dense.bias',(b0+Wo[:,drop]@imp2[f'E{L}'][drop]).astype(np.float32))\n    print(L,'dropped imp frac %.3f'%(np.sort(imp)[:3072-m].sum()/imp.sum()))\nprint(set_structure(layers=layers, ffn_keep=keep))"
}
```

