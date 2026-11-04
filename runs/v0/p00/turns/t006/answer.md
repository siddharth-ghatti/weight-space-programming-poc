## reasoning summary

I'm noting that "deductible" splits into several tokens, so I'll use the tokenizer on the full word set. The top 3000 tokens cover about 96.3% of practice text, and any unseen tokens at test time just get zero, so that's acceptable. I'm also considering layer 1 FFN neurons as extra detectors, which would give 6144 neurons aggregated in layer 2, possibly covering bigrams later. For now I'll focus on position sensitivity first.



## message

Keeping the top 3000 tokens costs almost nothing (96.3% CV). "deductible" isn't a single WordPiece, so I'll redo the position check with words that are single tokens.

## tool_use: run_python

```json
{
 "code": "E = base('bert.embeddings.word_embeddings.weight'); Pp = base('bert.embeddings.position_embeddings.weight'); T = base('bert.embeddings.token_type_embeddings.weight')\ng = base('bert.embeddings.LayerNorm.weight'); b = base('bert.embeddings.LayerNorm.bias')\ndef ln(x, gg, bb, eps=1e-12):\n    m = x.mean(-1,keepdims=True); v = ((x-m)**2).mean(-1,keepdims=True)\n    return (x-m)/np.sqrt(v+eps)*gg+bb\nDf = np.array([145,175,529,121,216,654,159,541])\ngam = np.ones(768,np.float32); gam[Df]=0\ndef h0(tokids, pos):\n    x0 = ln(E[tokids] + Pp[pos] + T[0], g, b)\n    return ln(x0, gam, np.zeros(768,np.float32))\nvv = vocab()\nfor w in ['premium','means','cancel','excluded','claim','the']:\n    k = vv[w]\n    hs = np.stack([h0(np.array([k]), p)[0] for p in [1,5,10,20,40,60]])\n    hn = hs/np.linalg.norm(hs,axis=1,keepdims=True)\n    print(w, np.round(hn@hn.T,3)[0])\n# cross-token cos at same position\nws = ['premium','means','cancel','excluded','claim','the','cancellation','covered','cover','covers']\nks = [vv[w] for w in ws]\nhs = h0(np.array(ks), 10); mu = h0(np.arange(1000,30522,7),10).mean(0)\nhc = hs-mu; hn = hc/np.linalg.norm(hc,axis=1,keepdims=True)\nprint(ws); print(np.round(hn@hn.T,2))"
}
```

