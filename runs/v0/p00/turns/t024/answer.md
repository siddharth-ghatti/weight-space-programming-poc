## reasoning summary

Everything lands in the 97.0–97.4% range, so the current blend stays. The differences are within noise. Next I want to be sure the implementation holds up end to end, so I'll go through each step on paper rather than simulate it, starting with the embedding in layer 0.

I'm stepping through the forward pass layer by layer on paper: the embedding, then self-attention with its zeroed output projection, then the FFN and the layer norms. In layer 1 the queries and keys vanish, so attention is uniform, and the value mean lands in the first eight dimensions before the output projection scatters it back.

Checking the pooler and classifier now: the pooled value for each class dimension is tanh of half the corresponding CLS feature, and the other dimensions sit at zero. The logits are just those pooled values, so the argmax is determined by the CLS features in the class dimensions. I also want to confirm the CLS feature in those dimensions is not picking up a shift that would change the ranking, and the common offset from the layer norm should cancel across the eight dimensions.

Checking the remaining edge cases now: the harness may cast my edits to float32, which should be fine. I also want to see whether the position-1 response is weaker and whether tokens at larger positions behave, since test sentences could run long. Tokens that appear in the test but not in practice, like near-duplicates of a known word, may pick up partial crosstalk, though that should stay semantically consistent.

I'm checking which unseen tokens get the largest outputs, since a max of 1.78 is high relative to the typical known-token scale. I want to see whether those are plausible near-duplicates of known words or just noise.

I'm checking which unseen tokens get the largest outputs and whether those are sensible near-duplicates of known words or just noise on common words.



All variants score about 97.0–97.4%, within noise, so I'm keeping the current blend. Next I'll check generalization on unseen tokens by listing the largest layer-0 outputs and the classes they push toward.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom lib import *\nem = Emb(base)\nL='bert.encoder.layer.%d.'\nWi=W(L%0+'intermediate.dense.weight'); bi=W(L%0+'intermediate.dense.bias'); W2=W(L%0+'output.dense.weight')\nK=np.load('K.npy'); S=np.load('S_final.npy')\ninv={i:t for t,i in vocab().items()}\nallv=np.arange(30522)\nH=em.h0(allv,15)\nout=np.zeros((30522,8),np.float32)\nfor s in range(0,30522,4000):\n    out[s:s+4000]=(gelu(H[s:s+4000]@Wi.T+bi)@W2.T)[:,DF]\nnp.save('out_vocab_p15.npy',out)\nunseen=np.setdiff1d(allv,K)\nnrm=np.linalg.norm(out,axis=1)\nprint('unseen out-norm pct', np.percentile(nrm[unseen],[50,90,99,99.9]), 'seen median', np.median(nrm[K]))\ntop=unseen[np.argsort(-nrm[unseen])[:60]]\nfor t in top:\n    print(f'{inv[t]:15s} {nrm[t]:.2f} -> {LABELS[out[t].argmax()]}')"
}
```

