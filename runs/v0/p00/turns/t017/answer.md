## reasoning summary

I'm tallying the 108 weight edits across the layers and the pooler and classifier, which comes to 108. Next I want to check the layer computations at the token level using the saved weights.

I'm moving on to check the layer-0 FFN output against the saved weights and comparing it to the centered target, then checking layer 1's value and output matrices and the pooler and classifier. I also need to update notes.md and look over a few places where Hugging Face's attention and activation handling could differ from my assumptions.

Next I'm checking the intermediate neuron outputs at a few token positions using the saved weights, comparing the result against the target.



All 108 tensors are saved, and I'm now checking the weights component by component: layer-0 FFN outputs, the LayerNorm gamma on the DF dims, layer-1 Q/K/V/O, and the pooler, classifier and no-op layers.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom lib import *\nem = Emb(base)\nS=np.load('S_final.npy')\nL='bert.encoder.layer.%d.'\ngam=W(L%0+'attention.output.LayerNorm.weight'); print('zeroed dims', np.where(gam==0)[0], 'DF', DF)\nWi=W(L%0+'intermediate.dense.weight'); bi=W(L%0+'intermediate.dense.bias'); W2=W(L%0+'output.dense.weight'); b2=W(L%0+'output.dense.bias')\ninv={i:t for t,i in vocab().items()}\nwords=['premium','means','cancel','excluded','claim','the','deduct','adjuster','renewal','arbitration','covered','not','.','[CLS]','[SEP]']\nvv=vocab()\nfor w in words:\n    k=vv[w]\n    for p in ([0] if w=='[CLS]' else [1,20,60]):\n        x0=ln(em.E[k]+em.P[p]+em.T[0],em.g,em.b)\n        h=ln(x0,gam,W(L%0+'attention.output.LayerNorm.bias'))\n        f=W2@gelu(Wi@h+bi)+b2\n        out=f[DF]\n        tgt=0.3*(S[k]-S[k].mean())\n        print(f'{w:12s} p{p:2d} out {np.round(out,2)} tgt {np.round(tgt,2)} nonDF max {np.abs(np.delete(f,DF)).max():.1e}')\nprint('L1 q,k max', np.abs(W(L%1+'attention.self.query.weight')).max(), np.abs(W(L%1+'attention.self.key.weight')).max(), np.abs(W(L%1+'attention.self.query.bias')).max(), np.abs(W(L%1+'attention.self.key.bias')).max())\nWv=W(L%1+'attention.self.value.weight'); print('Wv nz', [(int(i),int(j)) for i,j in zip(*np.nonzero(Wv))])\nWo=W(L%1+'attention.output.dense.weight'); print('Wo nz', [(int(i),int(j),float(Wo[i,j])) for i,j in zip(*np.nonzero(Wo))])\nWp=W('bert.pooler.dense.weight'); print('pool nz', [(int(i),int(j),float(Wp[i,j])) for i,j in zip(*np.nonzero(Wp))])\nWc=W('classifier.weight'); print('cls nz', [(int(i),int(j)) for i,j in zip(*np.nonzero(Wc))], W('classifier.bias'))\nfor l in range(1,12):\n    assert np.all(W(L%l+'output.dense.weight')==0) and np.all(W(L%l+'output.dense.bias')==0)\n    assert np.all(W(L%l+'attention.output.LayerNorm.weight')==1) and np.all(W(L%l+'output.LayerNorm.weight')==1)\n    assert np.all(W(L%l+'attention.output.LayerNorm.bias')==0) and np.all(W(L%l+'output.LayerNorm.bias')==0)\n    if l>=2: assert np.all(W(L%l+'attention.output.dense.weight')==0) and np.all(W(L%l+'attention.output.dense.bias')==0)\nprint('no-op checks passed')"
}
```

