## reasoning summary

Ruling out extra linear carries, so the depth stays at 9 content layers. Shrinking the model dimension looks like the better lever: dropping content and pool layers to m=768 would cut about 3.93M parameters, roughly 45%, and m=512 would go further. I'm now working out how to refit at 768 by reusing exposures 25 to 33 rather than running new passes.

I'm checking whether a smarter pruning method could beat energy-select plus ridge, given that the token data is limited to about 2000 rows from 64 sentences. Before committing to that, I want to estimate the accuracy hit from going down to m=768, since the margin over the 97.8 threshold could be only about 0.3 to 0.4 points.

Weighing whether a smaller model is worth the added risk: going to m=768 saves about 6% over the current size, but the margin over the 97.8 threshold is only around 0.3 to 0.4 points, and test-set sampling noise alone is roughly 0.36 points. I'm leaning toward keeping cross-validation accuracy at about 98.2 or higher, since failing the threshold outweighs a modest size gain. Improving the content-layer pruning fit might let m=768 hold accuracy, so I'm considering that route.

I'm weighing whether more token-level data could improve the pruning fit. The catch is that the post-attention-LayerNorm inputs I need aren't visible from hidden states alone, so the exposure approach is still required for each layer.

I'm weighing a different pruning route: keeping the FFN units but refitting the output weights to better match the full layer, since the post-attention inputs can only come from the exposure passes anyway. Only the 64 exposed sentences give usable inputs, so any refit at 768 has to work from that limited set.

I'm weighing whether the drift correction already covers the mismatch from using upstream inputs at m=768, which would make the 768 refit cost only the two exposure passes.

I'm planning the CV check on the M9 features with the pool anchors cut to 752 and to 496, since a big drop there would rule out m=768 outright. I'm also wondering what the 16 extra pool units (1024 minus 1008) are for, and whether they do anything beyond sitting unused.

I'm now checking the pool layer edits, specifically the intermediate weight rows kept for layer 11, to see how the 16 unused units are handled.



Extra linear carries don't help and slightly hurt, so depth stays at 9 content layers plus the pool layer. Next I'll check how the pool layer does with 752 or 496 anchors instead of 1008, before any m=768 refit.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nfrom lib7 import *\nF={k:v.astype(np.float64) for k,v in np.load('feats_seq1024_depth.npz').items()}\ny=labels_for([f'p{i:04d}' for i in range(1496)], practice, LABELS)\nfor na in [1008, 752, 624, 496]:\n    for lam in [0.02]:\n        r=[cv(F['M9'],F['C9'],F['M0'],y,c=10,lam=lam,c0=0.5,seed=sd,na=na) for sd in range(3)]\n        print(na,lam,np.round(r,4),round(np.mean(r),4), flush=True)\n# what is in pool layer 11 FFN kept units\nst=structure(); k11=np.array(st['ffn_keep'][11] if 11 in st['ffn_keep'] else st['ffn_keep']['11'])\nW1=W('bert.encoder.layer.11.intermediate.dense.weight')[k11]; b1=W('bert.encoder.layer.11.intermediate.dense.bias')[k11]\nnr=np.linalg.norm(W1,axis=1); print(len(k11), np.round(np.percentile(nr,[0,1,50,99,100]),4), np.round(np.percentile(b1,[0,50,100]),4))\nprint(np.where(nr<1e-6)[0][:20], (nr<1e-6).sum())\n"
}
```

