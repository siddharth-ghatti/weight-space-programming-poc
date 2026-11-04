## reasoning summary

I'm weighing whether combining the layer 0 bag-of-embeddings with a deep layer is worth it: it lifts accuracy from about 96% to 97.3%, roughly 13 more items out of 1024. Now I'm working out how to get mean(L0) plus mean(L9) into the final CLS, and I'm thinking about the residual stream in this post-LN BERT setup.

I'm looking at the post-LN BERT layers to see where the layer-0 mean could enter. Putting it in an early layer would spread through attention to all tokens, so I'm considering doing it at the final layer instead, where a uniform-attention head could pull in an approximation of the input information.

Next I'm checking whether the layer-0 mean could enter at the final layer instead, since CLS there only sees h11, which lacks exact input identity. Pooling into CLS at an earlier layer might disturb other tokens' attention, since CLS acts as a sink, but I could measure that.

I'm considering whether a deep-layer pooler could add capacity, since a ridge on the concatenated 1536-dim features might help even if the gain comes from weighting rather than new information. I'm also weighing a nonlinear route through the tanh pooler as random features, though the payoff there is uncertain, and I'm turning toward the final layer's FFN as a possibly better source of features.

I'm now looking at the FFN as a possible feature map: the output passes through a 3072-unit GELU layer, then LN2, which makes the final representation nonlinear and hard to fit in closed form, so I'm setting that aside. Attention-weighted pooling could help too, but a ridge on mean features is already strong. Given the CV ceiling of roughly 96–97%, I'm wondering whether the test set would behave similarly.

I'm checking a few alternatives before settling on a two-layer combination: mean-pooling the L9 states while excluding CLS and SEP, other layer pairs, and a shrinkage LDA classifier, with the full 1496-dimensional ridge to follow. The most feasible route to combining two layers looks like using the last two layers, but the problem is that CLS's state can't be isolated from the other tokens when the weights are shared.

I'm considering whether a head could be repurposed to give every token uniform attention in the final layer, but that would remove its original role for the other tokens and its output for CLS would just mix into the residual with everything else, which feels messy. The simplest robust option seems to be keeping uniform attention across all heads in layer 11.

Now I'm checking whether pooling the layer-9 or layer-8 mean into the final CLS would beat the layer-11 version, though those gains look within noise on 1024 samples. I'm also wondering if the layer-0 token-identity info can still reach CLS through h11, since its exact form is probably gone by then.

I'm checking whether the layer-0 mean adds anything over just more capacity, so I'll test the (0,11) and (0,12) combinations against a single layer with a higher regularization. I'm also considering whether x_cls at h11 could carry the mean of an earlier layer, though that would change how other tokens attend to CLS.

Now I'm weighing whether pooling into an earlier layer's CLS would disturb other tokens, since late-layer attention concentrates on [SEP] and punctuation and changing CLS would shift its key as well. Then I'm considering a cleaner idea: with uniform attention, every token would receive the same mean of the layer-10 states, so the pooling could be applied to the whole sequence at once.

I'm considering whether the layer-0 mean could be injected by making every token's state the same mean through uniform attention, so each token keeps its own identity plus the global average. Pooling again at the next layer would give CLS a nonlinear mix rather than a clean mean of the early states, so it's not obviously helpful. I'm now asking what actually differs between the layer-0 mean and the deeper layers: the first is a bag of word and position embeddings, while the deeper ones are contextual.

I'm weighing whether a tanh pooler could act as random nonlinear features for a ridge fit, which would only be a small step beyond the mean features. The sticking point is whether computing the pooler's tanh on the activations amounts to re-running the forward pass in numpy, which I'd rather avoid, so I'm considering reading the pooled output directly instead.

I'm settling on pooling from around layer 9 and planning to set up the layer 9 attention so CLS gets that mean, then checking the result by reading the pooled outputs from the edited model rather than simulating the forward pass in numpy.

I'm weighing whether to pool the layer-9 mean into the final CLS, since the differences from layer 11 look within noise. Splitting heads to read different inputs won't help, because they all read the same residual stream at that layer.

I'm weighing whether a head in layer 9 could be repurposed to pool into CLS through a subspace of 64 discriminative dimensions. But the same vector would also land in every other token's residual stream, which shifts their downstream states too, and BERT has no spare dimensions to absorb it. For about a 1% gain, that complexity doesn't seem worth it.

I'm checking whether the gain from layer 0 is real by re-running the (0,11) and (0,9) combinations under repeated folds, since the 9-item difference on 1024 samples could be noise. I'm also wondering whether attention weighted toward content words, rather than uniform pooling, would help, though the FFN route still looks unhelpful since it acts per token before any pooling.

I'm noticing that layer 10's feed-forward network works per token before the final layer pools, so I could redesign it to emit keyword-style features that the final mean-pool then carries to CLS. Linear token identity already survives a mean, so any real gain would have to come from the nonlinearity, such as sharper ReLU-like detectors.

I'm checking the cross-validation errors at around 96% to see whether they reflect label ambiguity that no feature could fix. If they look like noise, there's little left to gain from the layer-9 mean.



Mean-pooled deep layers reach about 96% on CV, and pairing layer-0 means with one lifts that to about 97.3%, though that is harder to wire into [CLS]. Next I'll check which sentences the deep-layer probe still misclassifies.



## tool_use: run_python

```json
{
 "code": "\nP = practice(); lab2i = {l:i for i,l in enumerate(LABELS)}\ny = np.array([lab2i[p['label']] for p in P])[:1024]\nrng = np.random.default_rng(0); perm = rng.permutation(1024); folds = np.array_split(perm, 5)\ndef ridge_pred(X, y, lam):\n    n,d = X.shape; pred = np.zeros(n,int)\n    for f in folds:\n        tr = np.setdiff1d(np.arange(n), f)\n        mu = X[tr].mean(0); sd = X[tr].std(0)+1e-6\n        Xt = (X[tr]-mu)/sd; Xv = (X[f]-mu)/sd\n        Y = np.eye(8)[y[tr]]\n        Wt = np.linalg.solve(Xt.T@Xt + lam*np.eye(d), Xt.T@(Y-Y.mean(0)))\n        pred[f] = (Xv@Wt).argmax(1)\n    return pred\nX = np.load('pass_2/mean_hidden_L11.npy').astype(np.float64)\npr = ridge_pred(X,y,100)\nprint('acc', (pr==y).mean())\nC = np.zeros((8,8),int)\nfor a,b in zip(y,pr): C[a,b]+=1\nprint(C)\nfor i in np.where(pr!=y)[0]:\n    print(f\"[{LABELS[y[i]]} -> {LABELS[pr[i]]}] ({P[i]['register']}) {P[i]['text']}\")\n"
}
```

