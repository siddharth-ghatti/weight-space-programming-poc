## reasoning summary

I'm working out importance per unit for each layer, using a normalized version of the hidden state as a stand-in for the FFN input. I'm also weighing whether using the next layer's hidden state as the proxy gives similar results, and I'll compare the two rankings to see if they agree.

I'm now planning to get each unit's importance by taking the variance of its GELU output under a Gaussian approximation, computed with Gauss-Hermite quadrature, then scaling by the norm of its output weights. I'm weighing that against empirical GELU statistics on the proxy pre-activations, which would be more accurate for heavy-tailed token distributions, but I'm leaning toward the Gaussian version because it's sufficient for ranking. I also want to check whether the final pooling layer's FFN output weights are zero.



I'm ranking the FFN units in each layer using two LN1-based proxies for robustness, scoring each by its GELU Gaussian-moment std times the output weight norm, and I'll fold dropped units' expected outputs into the bias.



## tool_use: run_python

```json
{
 "code": "from math import erf\nmask=np.load('pass_4/attention_mask.npy').astype(bool)\n# Gauss-Hermite for gaussian moments of gelu\ngh_x, gh_w = np.polynomial.hermite_e.hermegauss(40); gh_w=gh_w/gh_w.sum()\nfrom scipy.special import erf as serf\ngelu=lambda a: 0.5*a*(1+serf(a/np.sqrt(2)))\ndef gmoments(mu, sd):\n    A = mu[:,None]+sd[:,None]*gh_x[None,:]\n    G = gelu(A); E=(G*gh_w).sum(1); E2=(G**2*gh_w).sum(1); return E, np.sqrt(np.maximum(E2-E**2,0))\ndef norm_tok(H):\n    H=H-H.mean(-1,keepdims=True); return H/np.sqrt((H**2).mean(-1,keepdims=True)+1e-12)\nimp={}; Eg={}\nfor L in range(12):\n    g1=W(f'bert.encoder.layer.{L}.attention.output.LayerNorm.weight'); b1=W(f'bert.encoder.layer.{L}.attention.output.LayerNorm.bias')\n    Win=W(f'bert.encoder.layer.{L}.intermediate.dense.weight'); bin_=W(f'bert.encoder.layer.{L}.intermediate.dense.bias')\n    Wout=W(f'bert.encoder.layer.{L}.output.dense.weight'); on=np.linalg.norm(Wout,axis=0)\n    res_ = []\n    for src in [L, L+1]:\n        H=np.load(f'pass_4/token_hidden_L{src}.npy')[mask]\n        h=norm_tok(H)*g1+b1\n        A=h@Win.T+bin_\n        mu=A.mean(0); sd=A.std(0)\n        E,S=gmoments(mu,sd)\n        res_.append((E,S*on))\n    (E0,I0),(E1,I1)=res_\n    from scipy.stats import spearmanr\n    r=spearmanr(I0,I1).correlation\n    top0=set(np.argsort(-I0)[:768]); top1=set(np.argsort(-I1)[:768])\n    imp[L]=(I0+I1)/2; Eg[L]=(E0+E1)/2\n    print(L, 'spearman', round(r,3), 'top768 overlap', len(top0&top1), 'out-norm zero?', (on==0).sum(), 'imp share top768', round(np.sort(imp[L])[::-1][:768].sum()/imp[L].sum(),3))\nnp.savez(WORK/'ffn_imp.npz', **{f'imp{L}':imp[L] for L in range(12)}, **{f'E{L}':Eg[L] for L in range(12)})\n"
}
```

