## reasoning summary

I'm unexposing layer 8 and recording the map, then computing per-layer pruning for layers 0 through 8 using energy ranking and a least-squares refit on all tokens, with held-out errors as diagnostics. Next comes the evaluation pass with the pool layer, and I'm still working out what that layer should be, since features are read from the last content layer at index 9.

I'm weighing whether the pool layer is needed at all for screening, since the eval read can use just layers 0 through 8 and read the mean hidden state at layer 9. For the final model, I'm considering a pool layer with m units at the CLS position, and noting the free pooler's 768 tanh units as an alternative. I'm also deciding to pool only the KEEP dimensions and drop the reserved carry channels from the classifier to avoid leakage in cross-validation.

I'm planning to zero the carry heads' output weights for the reserved dimensions in layers 2 and 6, so those channels stay at zero and can't leak into the features, and I'm checking whether the layer norms' gains on those dimensions also need adjusting.

I'm now planning the pruning step: fit a least-squares refit for every layer with all tokens, and then write out the kept intermediate units and refit weights and bias for the chosen width.

I'm checking that the reserved rows of the refit stay at their original values, so I'll pin them there rather than let the refit move them. Then I'm running the m=1024, 1536, and 2048 analysis across all nine layers, measuring held-out error against both the F and S variances.



Layers 0–8 exposures are finished, so I'm restoring layer 8 and running the pruning analysis across all nine layers, with held-out error at four sizes to pick a ridge λ.



## tool_use: run_python

```json
{
 "code": "exec(open('lib.py').read()); exec(open('lib2.py').read())\njson.dump({'0':10,'1':11,'2':12,'3':13,'4':14,'5':15,'6':16,'7':17,'8':18}, open('expo_map.json','w'))\nunexpose(8, drop_edit); print(len(edits()))\nimport scipy.special as sp\ndef gelu(x): return 0.5*x*(1+sp.erf(x/np.sqrt(2)))\nemap=json.load(open('expo_map.json'))\nres={}\nfor l in range(9):\n    k=emap[str(l)]\n    Zp=np.load(f'pass_{k}/token_hidden_L{l+1}.npy'); msk=np.load(f'pass_{k}/attention_mask.npy').astype(bool)\n    Z=Zp[msk].astype(np.float64); sent=np.repeat(np.arange(64)[:,None],Zp.shape[1],1)[msk]\n    P=f'bert.encoder.layer.{l}.'\n    X=Z*base(P+'attention.output.LayerNorm.weight')+base(P+'attention.output.LayerNorm.bias')\n    W1=base(P+'intermediate.dense.weight').astype(np.float64); bb1=base(P+'intermediate.dense.bias')\n    W2=base(P+'output.dense.weight').astype(np.float64); bb2=base(P+'output.dense.bias')\n    A=gelu(X@W1.T+bb1); F=A@W2.T+bb2; S=X+F\n    tr=sent%2==0; te=~tr\n    Fden=np.linalg.norm(F[te]-F[te].mean(0))**2; Sden=np.linalg.norm(S[te]-S[te].mean(0))**2\n    en=A[tr].var(0)*(W2**2).sum(0)\n    out=[]\n    for m in [768,1024,1536,2048]:\n        keep=np.sort(np.argsort(-en)[:m])\n        As=A[:,keep]; mu=As[tr].mean(0); Fm=F[tr].mean(0); Ac=As[tr]-mu; Fc=F[tr]-Fm; W0=W2[:,keep].T\n        G=Ac.T@Ac; R=Ac.T@Fc\n        best=None\n        for lam in [30,100,300]:\n            Wn=np.linalg.solve(G+lam*np.eye(m), R+lam*W0); b=Fm-mu@Wn\n            e=np.linalg.norm(As[te]@Wn+b-F[te])**2\n            if best is None or e<best[0]: best=(e,lam)\n        out.append('%d:%.3f/%.4f(l%d)'%(m,best[0]/Fden,best[0]/Sden,best[1]))\n    print(l, '|F|/|S| var ratio %.2f'%(Fden/Sden), ' '.join(out))\n"
}
```

