## reasoning summary

I'm checking the ridge model at 3e-3, which lands around 97.1%, and the blend near 97.3%. I had been ignoring the intercept and length normalization, since the prediction is just a scaled version of the TF features. I'm working out what the intercept term would be when it's included.

I'm weighing whether to add the bias term. The CLS token's own output sits in the residual alongside the mean-pooled term, so it could act as a length-independent bias, but the standardization scaling makes that messy. I'll drop the bias, since it was ignored in the CV evaluation anyway, and recheck whether the blend or plain ridge at 3e-3 generalizes better.

I'm settling the blend-versus-ridge choice: the gap is small, so I'd rather keep the blend for its NB robustness than over-tune weights. Next I'm checking whether the token-level score approximations are accurate enough to trust the CV numbers, and I'm turning to the scale factor for the aggregated mean.

I'm sizing the output so the aggregated mean stays in a usable range: I'll look at the class-margin distribution from the scores table, then pick a scale G so that G·m sits around 1 to 5 against the CLS vector's norm. With the standardization constant around 760, the scaling barely changes sigma, and I'd want the pooler coefficient near 0.5 to keep the tanh inputs within ±2.

I'm checking that the layer-1 attention is effectively uniform, since with Q and K at zero every score is zero and only the pad mask changes anything. The later layers have their output projection zeroed, so their attention contributes nothing and the residual passes through standardization unchanged. Subtracting the mean in standardization is a common shift across all dimensions, so it doesn't change the argmax.

Checking the remaining BERT-specific details: the pooler reads the first token's hidden state, and the classifier maps that through an 8×768 weight matrix plus bias. Nothing relative-position or head-mask related is in play, since the model uses absolute positions.

Setting the head-0 value projection to pass through the matching input dimensions, and the attention output projection to scale them into the residual, both checked off. Then I'm writing the layer-0 feed-forward weights: the intermediate matrix carries the scaled detector directions with a bias that sets the threshold, and the output matrix maps each unit's activation back into the class dimensions with the fitted coefficients.

I'm checking that the layer-0 attention is harmless, since its output projection is zero and it adds nothing to the residual. Then I'm weighing whether the remaining capacity could go toward the tokens with no neuron, though the next piece only gets as far as layer 1's feed-forward block.

I'm weighing whether the spare capacity in layer 1's feed-forward block would help more than adding unigram detectors, since the 339 neuron-less tokens rarely show up in test sentences. Moving aggregation to layer 2 would free that block, but the gain looks marginal, so I'm considering bigram features as an alternative, which would need attention I don't have yet.

I'm weighing whether a previous-token head could work in layer 1: attention would need to peak at the preceding position, and fitting separate query and key projections by least squares looks shaky with token content interfering. The expected gain is only about 1%, so I'll set that aside, build and save the unigram model first, and revisit enhancements afterward.

I'm now looking at how to pick the scale so the blended scores land near 2 at the 99th percentile, and how to weight the fit for the tokens that have neurons.

I'm setting the special tokens' weight to 200 and extending the position grid out to about 80, so the detectors stay reliable if a test sentence runs longer than the 63-token maximum I've seen so far.



Ridge-TF and the NB/4 + ridge blend both score about 97% in cross-validation, and I'm building the build script around the blend fitted on all 1496 sentences: computing the token table, picking the 3072 most informative tokens, building the whitened detectors, fitting weighted ridge output weights, and reporting fit quality.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom lib import *\nem = Emb(base)\nCb=np.load('Cb.npy'); Cw=np.load('Cw.npy'); mu=np.load('mu.npy'); K=np.load('K.npy')\nP,y,ids,X = load_data(practice, tokenize, LABELS)\nV=30522; N=len(P)\nn=np.array([len(t) for t in ids]); Xtf=sp.diags(1.0/n)@X\ncnt=np.zeros((V,8))\nfor c in range(8): cnt[:,c]=np.asarray(X[y==c].sum(0)).ravel()\nlp=np.log(cnt+0.3)-np.log(cnt.sum(0)+0.3*V); Snb=lp-lp.mean(1,keepdims=True)\nYc=np.eye(8)[y]; Kk=(Xtf@Xtf.T).toarray(); A=np.linalg.solve(Kk+3e-3*np.eye(N), Yc-Yc.mean(0))\nSr=np.asarray(Xtf.T@A); Sr-=Sr.mean(1,keepdims=True)\nS = Snb/4 + Sr\nS[[0,101,102]] = 0\nseen = cnt.sum(1)>0\nS[~seen]=0\nprint('train acc of table (sum):', ((X@S).argmax(1)==y).mean())\ntot=cnt.sum(1)\nnr=np.linalg.norm(S[K],axis=1); print('||S|| pct', np.percentile(nr,[50,90,99,100]))\nnp.save('S_final.npy', S.astype(np.float32))\ninfo=tot[K]*np.linalg.norm(S[K],axis=1)\nKn=K[np.argsort(-info)[:3072]]; np.save('Kn.npy',Kn)\nprint('neuronless count', tot[np.setdiff1d(K,Kn)].sum())\nMk=sum(em.h0(Kn,p) for p in [3,15,35])/3\nC=Cb+3*Cw+0.05*np.eye(768)\nU=np.linalg.solve(C,(Mk-mu).T).astype(np.float32)\nU/=np.einsum('dk,kd->k',U,(Mk-mu))[None,:]\nprint('U[DF] max', np.abs(U[DF]).max(), 'U norms', np.percentile(np.linalg.norm(U,axis=0),[1,50,99]))\nnp.save('U.npy',U)\nbeta,theta=10.0,0.6\nrng=np.random.RandomState(1)\nrand=np.setdiff1d(rng.choice(V,4000,replace=False),K)\nsamp=np.concatenate([K,rand]); wts=np.concatenate([tot[K]+1.0,0.5*np.ones(len(rand))]); tgt=S[samp]\ngrid=[1,2,3,4,6,8,11,14,18,22,27,32,38,44,51,58,65,75]\nG=np.zeros((3072,3072)); B=np.zeros((3072,8))\nfor p in grid:\n    Af=gelu(beta*((em.h0(samp,p)-mu)@U-theta)).astype(np.float32)\n    Aw=Af*wts[:,None]; G+=Af.T@Aw; B+=Aw.T@tgt\nspec=np.concatenate([em.h0(np.array([101]),0)]+[em.h0(np.array([102,0,100]),p) for p in grid])\nAsp=gelu(beta*((spec-mu)@U-theta)); G+=200*Asp.T@Asp\nlam=1e-2*np.trace(G)/3072\nO=np.linalg.solve(G+lam*np.eye(3072),B)\nnp.save('O.npy',O.astype(np.float32))\n# eval\nerrK=[];outR=[]\nfor p in grid:\n    Af=gelu(beta*((em.h0(samp,p)-mu)@U-theta)); pr=Af@O\n    errK.append(((pr[:len(K)]-tgt[:len(K)])**2).sum(1)); outR.append(np.linalg.norm(pr[len(K):],axis=1))\nerrK=np.array(errK); outR=np.array(outR)\nw=tot[K]\nprint('weighted rel err', (errK.mean(0)*w).sum()/((nr**2)*w).sum())\ninK=np.isin(K,Kn)\nprint('rel err neuron tokens median', np.median(errK.mean(0)[inK]/nr[inK]**2), '99pct', np.percentile(errK.mean(0)[inK]/nr[inK]**2,99))\nprint('rand out norm pct', np.percentile(outR,[50,90,99,100]))\nprint('special outs', np.abs(Asp@O).max())"
}
```

