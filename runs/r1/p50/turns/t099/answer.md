## reasoning summary

I'm weighing a small weighted-fit gain in L3 and L4 against small losses in the lower levels, so the plan is to use var selection for L0 to L2 and g2sig for L3 and L4, with the weighted fit applied to L3 and L4 and lambda at 0.1. Summed up, L3 drops about 10% and L4 about 22% in error energy, which might be roughly 7% less total noise overall. Now I'm checking why var versus g2sig matters in L4, since tokens with small sigma get amplified there and I want to see which

I'm now wondering which tokens have small σ in L4 — probably special tokens like </s> or punctuation acting as attention sinks, where errors could get amplified by LayerNorm. Since the CLS query in layer 5 is what really matters, token importance for L4 errors should scale with the CLS attention weights on each token, which I could read from the attention maps.

I'm weighing whether attention from the layer-5 CLS token should set how much each token counts in the L4 fit. Weighting purely by attention would concentrate the fit on a few tokens and hurt generalization, so a mixed weight that blends in a uniform term seems safer. Before committing, I'd like to know how much of the noise actually comes from L4 compared with the other layers.

I'm weighing two plans: a conservative final run at m=1536 using the improved selection and fits, accepting it if agreement holds at or above 0.981, or exploring m=1280 with the same fits and comparing the two. I'm also settling the data question: the existing B1 and B2 batches came from an m=1024 upstream, which is more perturbed than an m=1536 one, but they still cover the edited distribution reasonably, so I'll train on everything I already have. Then I'm wondering how much the practice-wrong items might predict test-time flips, using them as a proxy for hard cases.

I'm sizing up whether the noise margins from practice will hold at the candidate setting, and I'm thinking of modeling test flips as a sum over the probability that noise exceeds each item's margin. The hard part is guessing the test margin distribution, since I'd have to deflate the in-sample logits to get it. Given how uncertain that modeling is, I'm leaning on the earlier threshold: keeping test flips under about 2% seems like the working target.

I'm checking whether the governance false positives drive the macro-F1 drop by comparing the edited confusion matrix against the base one. I'm also weighing a target of m=1536 with the improved fits, possibly testing m=1280 too.

I'm checking whether the fit targets should match the full FFN output on the edited inputs, or instead match the base layer's output on base inputs so that upstream errors get corrected as well. I'm also noting that layer 0's inputs are exact under any pruning config, so its data is clean.

I'm now asking whether paired data would make an error-correcting target possible, since the edited and base FFN inputs for the same tokens were never collected together. The earlier end-to-end test at L5 suggested correction wouldn't help at CLS, but token-level correction might, because some residual error depends on each token's own input.

I'm considering whether paired base and edited FFN inputs would allow an error-correcting fit, which would take about four more passes for layers 1 to 4. I doubt upstream error is predictable from the local input alone, since it depends on context through attention, so I'm leaning toward skipping this.

I'm going with the improved fits now, and I'll evaluate m=1536 and m=1280 in two passes. Before that, I'll write a new fit2.py with a helper that takes a layer, the m setting, lambda, and a selection mode of either var or g2sig, plus a weighted option. For L5, I'll use the CLS data from the m=1024 upstream together with the base data, with lambda at 0.1 and var selection, since the CV already showed L5 fits fine.



The weighted fit gives a small gain on L3/L4 and a slight loss on L0–2, so I'll use variance ranking unweighted for L0–2 and g2sig with the 1/σ²-weighted fit for L3–4. I'll write that as fit2.py and rebuild the m=1536 model, plus m=1280 under a separate keep-set file.



## tool_use: run_python

```json
{
 "code": "code = r'''\nimport json\nexec(open(WORK/'seq.py').read())\ndef fit_layer2(L, m, lam=0.1, sel=None, weighted=None, save=True):\n    sel = sel or ('var' if L<3 else 'g2sig'); weighted = (L>=3) if weighted is None else weighted\n    pre=f\"roberta.encoder.layer.{L}.\"\n    W1=base(pre+\"intermediate.dense.weight\").astype(np.float64); b1=base(pre+\"intermediate.dense.bias\")\n    W2=base(pre+\"output.dense.weight\").astype(np.float64); b2=base(pre+\"output.dense.bias\"); g2=base(pre+\"output.LayerNorm.weight\").astype(np.float64)\n    A,_=all_data(L); A=A.astype(np.float64)\n    H=gelu(A@W1.T+b1); U=A+H@W2.T+b2; sig=U.std(1,keepdims=True); w=1/sig**2\n    if sel=='var': sc=H.var(0)*(W2**2).sum(0)\n    else: sc=((H-H.mean(0))**2*w).mean(0)*((g2[:,None]*W2)**2).sum(0)\n    keep=np.sort(np.argsort(-sc)[:m]); drop=np.setdiff1d(np.arange(3072),keep)\n    T=H[:,drop]@W2[:,drop].T; X=H[:,keep]\n    ww = w if weighted else np.ones_like(w)\n    mu=(X*ww).sum(0)/ww.sum(); mt=(T*ww).sum(0)/ww.sum(); Xc=X-mu; G=Xc.T@(Xc*ww)\n    B=np.linalg.solve(G+lam*np.trace(G)/m*np.eye(m), (Xc*ww).T@(T-mt))\n    if save:\n        W2n=np.zeros_like(W2); W2n[:,keep]=W2[:,keep]+B.T\n        save_edit(pre+\"output.dense.weight\", W2n.astype(np.float32))\n        save_edit(pre+\"output.dense.bias\", (b2+mt-mu@B).astype(np.float32))\n    return keep.tolist()\ndef fit_L5b(m, lam=0.1, save=True):\n    pre=\"roberta.encoder.layer.5.\"\n    W1=base(pre+\"intermediate.dense.weight\").astype(np.float64); b1=base(pre+\"intermediate.dense.bias\")\n    W2=base(pre+\"output.dense.weight\").astype(np.float64); b2=base(pre+\"output.dense.bias\")\n    A=np.r_[np.load(WORK/'ffn_ed_L5_cls.npy'), np.load(WORK/'ffn_in_L5_cls.npy')].astype(np.float64)\n    H=gelu(A@W1.T+b1); sc=H.var(0)*(W2**2).sum(0); keep=np.sort(np.argsort(-sc)[:m]); drop=np.setdiff1d(np.arange(3072),keep)\n    T=H[:,drop]@W2[:,drop].T; X=H[:,keep]; mu=X.mean(0); mt=T.mean(0); Xc=X-mu; G=Xc.T@Xc\n    B=np.linalg.solve(G+lam*np.trace(G)/m*np.eye(m), Xc.T@(T-mt))\n    if save:\n        W2n=np.zeros_like(W2); W2n[:,keep]=W2[:,keep]+B.T\n        save_edit(pre+\"output.dense.weight\", W2n.astype(np.float32))\n        save_edit(pre+\"output.dense.bias\", (b2+mt-mu@B).astype(np.float32))\n    return keep.tolist()\ndef build2(m, lam=0.1, keepfile=None):\n    ks={}\n    for L in range(5): ks[L]=fit_layer2(L,m,lam)\n    ks[5]=fit_L5b(m,lam)\n    if keepfile: json.dump({str(k):v for k,v in ks.items()}, open(WORK/keepfile,'w'))\n    return set_structure(layers=list(range(6)), ffn_keep=ks)\n'''\nopen('fit2.py','w').write(code)\nexec(open('fit2.py').read())\n# backup current m1536 edits\nimport shutil\nif not os.path.exists('edits_m1536_v1'): shutil.copytree('edits','edits_m1536_v1'); shutil.copy('structure.json','structure_m1536_v1.json')\nprint(build2(1536, keepfile='keep_m1536_v2.json'))\nprint(structure()['m'], edits()[:3] if isinstance(edits(),list) else list(edits())[:3])\n"
}
```

