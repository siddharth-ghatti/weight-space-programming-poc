## reasoning summary

The QKV read matrices have a smallest singular value only about 3x below their median, and the FFN read matrix W1 is similarly well-conditioned, so none of them look strongly inert. W_o has a few very small singular values, meaning it writes little along some directions, but the read side overall is close to full-rank, and combining directions across layers would likely be worse, so I'm heading toward the conclusion that

I'm weighing a fallback: accept a moderate perturbation by carrying a few modest-magnitude numbers, though the write-side corruption from W2 is also a concern. Then I'm shifting to a different design, where I'd decode per-token identities from the deep states at the end and apply the shallow bag-of-words scoring to those, approximating the token-level mapping with a linear regression.

I'm thinking through how to decode per-token identities with a nonlinear step, since a linear map on the mean just reproduces the failed ridge fit. The plan is an FFN on layer 10's token states, with the output pooled at the CLS position in layer 11. The open question is how to fit that per-token decoder in closed form.

I'm trying a per-token nonlinear decoder instead, since a linear map on the mean just repeats the failed ridge fit. The idea is to freeze layer 10's attention, pass the token states through its FFN as random GELU features, and fit a ridge readout from those features to each token's projected embedding, with the token-level data coming from about 2000 tokens per pass.

I'm estimating how well per-token decoding would recover the shallow targets: if each token is decoded with moderate accuracy, the errors should partly cancel when averaged over roughly thirty tokens per sentence. The final score would then be the deep-layer projection plus that averaged shallow estimate.

I'm wondering whether token identity can even be decoded nonlinearly from layer 10 with a random-feature model over roughly 6000 tokens, since the vocabulary is large and overfitting is a risk, though ridge should help. The fact that the shallow layer adds information beyond layer 11 might mean exact word identity gets blurred in deeper layers, so if the decoder recovers the shallow targets well, that information was present all along, but if it only reaches an R² near 0.5, the approach stalls.

Token-level regression has far more samples than the sentence-level fit, since 40k tokens against 1196 sentences, so the linear map from layer 10 to the shallow targets should be much better estimated. That would make the combined classifier a simple linear function of M10, with weights (A10 + A0·R) applied to the pooled states.

I'm checking whether token-level linear regression could even work here: a linear map from layer 11 token states to the layer-0 targets, fit on roughly 8000 tokens across a few passes, which is a much larger sample than the sentence-level fit. The open question is whether that signal is linearly recoverable from the deep states.

I'm considering whether the layer-0 targets could be read from token-level data instead of the sentence-level fit, since 40k tokens give far more samples than 1196 sentences, so a linear per-token map from layer 11 might be estimated much better. I'd still have to read layer 0 states from the passes, because rebuilding the embedding layer by hand isn't allowed. A nonlinear per-token decoder through layer 10's FFN remains a possibility, but I'll only pursue it if the linear result falls short.

I'm considering a token-level ridge from each token's layer-11 state to the sentence label, averaged per sentence. That amounts to a linear classifier on the pooled states with a different regularizer, weighted by token count and using the within-sentence covariance, which could generalize better. I'll test it with token data from layers 0, 10, 11 and 12, collected across the 64 texts per pass.

Planning how many passes to spend on token-level reads, starting with two passes to see if a token-level linear map from the deep states improves the sentence-level fit.

I'm weighing whether the token-level linear map could work better than the sentence-level fit: with thousands of tokens instead of 1196 sentences, the regression should be far better estimated. If it reaches a high R², the reconstruction of the shallow states from the deep ones might hold up after averaging per sentence, and I'd then just need a classifier on the layer-11 states plus that reconstruction. I'm also considering fitting the shallow scores directly rather than through the full reconstruction.

I'm weighing whether a nonlinear token-level decoder could be built by repurposing layer 10's FFN, but that would mean re-deriving the layer norm and feed-forward step by hand from the read activations, which the rules rule out.

I'm moving on to planning the pass budget: two passes to pull token-level states from layers 0, 11 and possibly 12 for about 128 texts. I'm also settling on the cleaner route for the classifier, which is to read the edited model's actual pooled outputs after the architectural change and fit on those, rather than simulating layer norm by hand.

I'm settling on the edit where the layer 11 mean-pooled state becomes the CLS output, which has the best accuracy at 95.9%. The plan is two passes to read the edited CLS hidden state for all 1496 texts, fit the classifier on those, and use one more pass to check the predictions.

I'm considering whether the CLS state could mix the two options: half the attention heads pool uniformly over layer 11, while the other heads keep their base CLS-specific attention. That might recover the base model's CLS information without giving up the pooling gain.

I'm considering whether the 768-dimensional attention output is effectively an invertible linear transform of the concatenated head outputs, so that if the attention contribution is scaled up, the CLS state would be roughly a normalized version of that transform and the residual would matter little.

I'm thinking about whether the CLS state can hold a concatenation of differently-attending heads, six uniform ones projecting the mean of layer 11 and six keeping base attention, since the output projection makes the 768-dim attention output an invertible mix of the head channels.

I'm going to pull token-level states from layers 0, 6 and 11 over two passes, using 64 texts stratified by label, so I can test whether a token-level linear map or classifier does better than the sentence-level fit.



The residual stream has no inert subspace to carry a signal through, so I'm dropping that path. I'll test token-level regression from layer 11 to layer 0 instead, using 2 read passes on 128 stratified texts at layers 0, 6 and 11.



## tool_use: run_python

```json
{
 "code": "import json\nP=practice(); lab={l:i for i,l in enumerate(LABELS)}\nrng=np.random.RandomState(42)\nsel=[]\nfor c in LABELS:\n    idx=[i for i,p in enumerate(P) if p['label']==c]; rng.shuffle(idx); sel+=idx[:16]\nsel=sorted(sel)\nids=[P[i]['id'] for i in sel]\njson.dump({'A':ids[:64],'B':ids[64:]}, open('tok_ids.json','w'))\nprint(json.dumps(ids[:64])); print(json.dumps(ids[64:]))"
}
```

