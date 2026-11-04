## reasoning summary

I'm moving on to the covariance and detector step now. The token set is every token in the practice data that appears at least once, minus the special [CLS] and [SEP] tokens, which comes to roughly 3,400 tokens. I'm weighing whether to draw on the layer 0 feed-forward neurons alone or also on layer 1, and I'm planning to check selectivity first.

I'm building the covariance matrix from two parts: between-token variation of averaged layer 0 activations across the vocabulary, and within-token variation sampled over positions, plus a ridge term. The detector for each token will be the inverse covariance applied to its mean deviation from the overall mean, and I'll judge it by self response against the maximum cross response from other tokens.



I'm computing between-token and within-token covariances, then building whitened detectors per practice token to check their selectivity against the strongest competing vocab token.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom lib import *\nimport time\nem = Emb(base)\nP,y,ids,X = load_data(practice, tokenize, LABELS)\ntot = np.asarray(X.sum(0)).ravel()\nK = np.array([k for k in np.argsort(-tot) if tot[k]>0 and k not in (101,102)])\nprint('K size', len(K))\nallv = np.arange(30522)\nt0=time.time()\nmpos = [3, 15, 35]\nM = np.zeros((30522,768),np.float32)\nfor p in mpos: M += em.h0(allv,p)/len(mpos)\nmu = M.mean(0)\nCb = np.cov((M-mu).T)\n# within: random tokens + K tokens, positions 1..70\nrng=np.random.RandomState(0)\nsamp = np.concatenate([rng.choice(30522,1500,replace=False), K[:1500]])\npositions = np.arange(1,71)\nH = np.stack([em.h0(samp,p) for p in positions],1)  # [n,70,768]\nD = (H - H.mean(1,keepdims=True)).reshape(-1,768)\nCw = D.T@D/len(D)\nprint('time',time.time()-t0, 'tr Cb',np.trace(Cb),'tr Cw',np.trace(Cw))\nnp.save('Cb.npy',Cb); np.save('Cw.npy',Cw); np.save('mu.npy',mu); np.save('K.npy',K)\nev = np.linalg.eigvalsh(Cb); print('Cb eig range', ev[:3], ev[-3:])\nev = np.linalg.eigvalsh(Cw); print('Cw eig range', ev[:3], ev[-3:])"
}
```

