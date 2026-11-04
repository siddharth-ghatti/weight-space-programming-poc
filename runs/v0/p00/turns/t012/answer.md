## reasoning summary

Bigrams look like only about a 1% gain, so they're probably not worth the risk right now, maybe later if there's time. The bigger concern is implementing without any verification, so I'll focus on a careful unigram version first and consider extras afterward. Next I'm working through the full pipeline layer by layer, checking each component analytically rather than simulating everything end to end, starting with layer 0's attention projections.

I'm setting up the layer 0 weights: zeroing the attention output projection so the block passes the token embedding through unchanged, and building the intermediate and output FFN weights to write a signal into the reserved dimensions. I'm also checking how the centering of the LayerNorm affects the mean once the FFN contribution is added.

I'm working out how the layer 1 mean and variance come out once the layer 0 contribution is included: the shift in the mean is tiny and the same across all eight class dimensions, so it doesn't change the argmax. The normalized class signal is then (s_t − mean_t)/σ_t. Next I'm setting up layer 1's attention so the query and key weights are zero, which makes the attention weights uniform, and starting on the value projection.

I'm setting the layer 1 value projection so the first eight value dimensions carry the normalized class signal, then averaging those across tokens into the CLS context. Next I'm writing the attention output matrix to copy that context back into the class dimensions, scaled by G, and checking that the residual leaves the CLS class values near zero with only a tiny common shift.

Still writing the layer-by-layer weight setup...

Still writing the pooler and classifier weights...

Checking the per-token score magnitudes next, since the classifier's tanh saturation depends on how large the NB sums get.

I'm checking that the scores stay small enough that the per-token normalization barely shifts, so scaling them down by about 0.3 should keep the layer-1 variance nearly constant across tokens.

I'm checking that the layer 1 attention targets for CLS and SEP positions, along with any PAD tokens, are zero so they don't contribute to the score.

I'm setting up the least-squares fit for the output weights, using the known tokens plus the CLS, SEP, and PAD positions as samples. The features are the neuron activations at each sampled position, and I'm weighing whether to keep every position or average over them to keep the matrix small.

Next I'm settling how to choose which tokens get neurons, weighing raw frequency against frequency times score magnitude, since frequent tokens with flat class profiles still add up across many occurrences.

I'm deciding which tokens get dedicated neurons, ranking them by count times score magnitude. For the tokens left without a neuron, I'm weighing whether to fit them to their true targets with low weight or to zero, since a conflict with similar neuron-backed tokens could distort the fit.

I'm weighing each token's sample weight by its occurrence count, maybe with sqrt to keep "the" from dominating, since errors on frequent tokens add up across many sentences. I'm also considering adding a few thousand unseen vocab tokens with target zero at low weight, to limit spurious scores, though that could cost the semantic generalization I'd hoped for.

I'm checking the random vocabulary tokens as low-weight zero targets to keep spurious outputs small, and I want to see whether related unseen words still get reasonable scores. Next I'll pick β and θ by evaluating how well the ridge fit reproduces the targets for a few settings.

I'm weighing whether to evaluate the fit by reimplementing layer 0's forward pass on practice sentences, which seems like simulating the model, so I'll drop that. Instead I'll measure token-level error of the predicted scores against the targets across vocab rows, and test how noise of the same size as the fit residual shifts the NB decisions.

Checking whether a token-level evaluation of the fitted scores counts as running the network; I'm treating it as a per-token check of the design, not a full simulation, so I'll go ahead and build the neuron activations and fit the output weights.

Next I'm sizing the hidden activations at each position, which comes to roughly 136MB for the token embeddings and about 10.5M values per position for the neuron pre-activations, so it fits in memory. I'm also weighing the GELU negative tail for non-matching tokens: with a large β and θ around 0.5, those pre-activations land near −5 and contribute almost nothing, while neurons with moderate responses near −2 still add a small amount that the fit can account for. Next I'll try a few β and θ combinations and compare their fit residuals.



Bigrams would add only about a point and need a fragile attention head, so I'm skipping them in favour of a clean unigram build. Next I'll fit the layer-0 FFN's output weights by weighted ridge and compare a few (β, θ) settings by token-level fit error.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom lib import *\nimport time\nem = Emb(base)\nCb=np.load('Cb.npy'); Cw=np.load('Cw.npy'); mu=np.load('mu.npy'); K=np.load('K.npy')\nP,y,ids,X = load_data(practice, tokenize, LABELS)\nV=30522; alpha=0.3\ncnt=np.zeros((V,8))\nfor c in range(8): cnt[:,c]=np.asarray(X[y==c].sum(0)).ravel()\nlp=np.log(cnt+alpha)-np.log(cnt.sum(0)+alpha*V); S=lp-lp.mean(1,keepdims=True)\nS[[101,102,0]]=0\ntot=cnt.sum(1)\ninfo = tot[K]*np.linalg.norm(S[K],axis=1)\nKn = K[np.argsort(-info)[:3072]]   # neuron tokens\nprint('tokens without neuron:', len(K)-3072, 'their total count', tot[np.setdiff1d(K,Kn)].sum(), 'of', tot[K].sum())\nnp.save('Kn.npy',Kn); np.save('S_nb.npy',S.astype(np.float32))\nMk = sum(em.h0(Kn,p) for p in [3,15,35])/3\nC = Cb + 3*Cw + 0.05*np.eye(768)\nU = np.linalg.solve(C, (Mk-mu).T).astype(np.float32)\nU /= np.einsum('dk,kd->k',U,(Mk-mu))[None,:]\nnp.save('U.npy',U)\nrng=np.random.RandomState(1)\nrand = np.setdiff1d(rng.choice(V,3000,replace=False), K)\ngrid=[1,2,3,5,8,12,17,23,30,38,47,57,64]\nsamp = np.concatenate([K, rand])\nwts = np.concatenate([tot[K]+1.0, 0.5*np.ones(len(rand))])\ntgt = S[samp]\nfor beta,theta in [(10,0.5),(10,0.6),(20,0.6),(6,0.4)]:\n    G = np.zeros((3072,3072)); B = np.zeros((3072,8))\n    feats=[]\n    for p in grid:\n        A = gelu(beta*((em.h0(samp,p)-mu)@U - theta)).astype(np.float32)\n        Aw = A*wts[:,None]\n        G += A.T@Aw; B += Aw.T@tgt\n        feats.append(A)\n    # special tokens: CLS@0 , SEP/PAD at grid\n    sp_samples=[em.h0(np.array([101]),0)]+[em.h0(np.array([102,0]),p) for p in grid]\n    Asp = gelu(beta*((np.concatenate(sp_samples)-mu)@U-theta))\n    G += 200*Asp.T@Asp\n    lam=1e-2*np.trace(G)/3072\n    O = np.linalg.solve(G+lam*np.eye(3072), B)\n    errs=[]; \n    for A in feats:\n        pred=A@O; errs.append(((pred-tgt)**2).sum(1))\n    errs=np.array(errs)  # [grid, samp]\n    nK=len(K)\n    relK = errs[:,:nK].mean(0)/ (np.linalg.norm(tgt[:nK],axis=1)**2+1e-9)\n    w=tot[K]\n    print(f'beta={beta} th={theta}: weighted rel err K {np.sum(errs[:,:nK].mean(0)*w)/np.sum((np.linalg.norm(tgt[:nK],axis=1)**2)*w):.4f}, median rel {np.median(relK):.4f}, rand mean sq out {errs[:,nK:].mean():.4f}, pos1 vs pos30 err {errs[0,:nK].mean():.4f} {errs[8,:nK].mean():.4f}, special out {np.abs(Asp@O).max():.4f}, maxabs O {np.abs(O).max():.2f}')"
}
```

