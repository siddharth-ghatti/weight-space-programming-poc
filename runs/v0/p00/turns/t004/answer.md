## reasoning summary

The unigram Naive Bayes baseline is already at about 96% cross-validation accuracy, so a bag-of-wordpieces classifier should be enough. Next I need to implement it faithfully inside the network. The ridge model's bias gets added after the mean of the token scores, which is what the network already computes, so that part matches.

Now I'm weighing how sensitive the classifier is to detector errors, especially tokens that are unseen at test time. Unseen tokens should contribute nothing after centering. I'm also checking how many tokens need detectors, since layer 0 has 3072 neurons against roughly 3413 tokens, so I'd need to prune to the most frequent ones and test the accuracy cost.

I'm now going to look at the embedding weights and the layer norm parameters, so I can compute the layer-0 hidden states for each vocabulary token at a range of positions. Holding the full set of positions in memory would be too large, so I'll sample a handful of positions instead, and then estimate a per-token average to see how the tokens separate.

Next I'm setting up the detector response: for each vocabulary token I'll project its layer-0 hidden state onto a direction that picks out that token, check its self-response against the cross-response from other tokens, and then fit the output weights from the resulting neuron activations.

Next I'm weighing how to choose the eight extra dimensions, looking for ones with low variance across vocabulary tokens and positions so little information is lost, and avoiding the outlier dimensions that show up in the layer-norm output.

Now I'm checking how the layer-0 and layer-1 attention reads and writes the feature dimensions, so the CLS token ends up carrying the detector signal.

Now I'm settling on the scale of the token scores: the centered log-probabilities run roughly from -3 to 5, which keeps the per-token spread near 1, and scaling them down a bit might reduce the distortion from detector errors.

I'm working out how much a gelu tail from non-matching detectors could matter: with a sharp beta, the negative values for tokens that barely match come out small, and the least-squares output fit should absorb most of that crosstalk for the tokens I already know.

Checking the embedding layer norm parameters and the position range next, to see how the layer-0 hidden states look across the vocabulary.



Unigram bag-of-wordpieces reaches about 96% CV accuracy, so a keyword-detector network should work. I'm checking the embedding-layer weights and picking 8 low-information dims to reserve as feature dims next.



## tool_use: run_python

```json
{
 "code": "E = base('bert.embeddings.word_embeddings.weight')\nPp = base('bert.embeddings.position_embeddings.weight')\nT = base('bert.embeddings.token_type_embeddings.weight')\ng = base('bert.embeddings.LayerNorm.weight'); b = base('bert.embeddings.LayerNorm.bias')\nprint('E norm', np.linalg.norm(E,axis=1).mean(), 'P norm', np.linalg.norm(Pp[:80],axis=1).mean(), 'T norm', np.linalg.norm(T,axis=1))\nprint('gamma stats', g.min(), g.max(), np.sort(np.abs(g))[-5:], np.argsort(np.abs(g))[-5:])\nprint('beta stats', b.min(), b.max(), np.argsort(np.abs(b))[-5:])\ndef ln(x, gg, bb, eps=1e-12):\n    m = x.mean(-1,keepdims=True); v = ((x-m)**2).mean(-1,keepdims=True)\n    return (x-m)/np.sqrt(v+eps)*gg+bb\n# x0 for all vocab at position 10\nx0 = ln(E + Pp[10] + T[0], g, b)\nz = (x0 - x0.mean(-1,keepdims=True))/x0.std(-1,keepdims=True)\nprint('per-dim std of standardized x0 across vocab: min', z.std(0).min(), 'max', z.std(0).max())\nprint('per-dim mean abs large dims', np.argsort(np.abs(z.mean(0)))[-6:], np.sort(np.abs(z.mean(0)))[-6:])\norder = np.argsort(z.std(0))\nprint('lowest-var dims', order[:12], z.std(0)[order[:12]], np.abs(z.mean(0))[order[:12]])"
}
```

