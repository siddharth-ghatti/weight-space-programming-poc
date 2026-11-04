## reasoning summary

I'm checking that z has zero mean and unit variance per token, then moving on to the layer 0 FFN activations: computing the GELU pre-activation, the contribution energies, and the least-squares reconstruction error against m. I also want to check the full FFN reconstruction against the base token_hidden L1, but I don't have base L1 for these token ids, so that comparison would need a separate self-consistency check.

I'm planning the layer 0 pruning analysis: rebuilding the FFN input from the LN1 parameters, computing the full FFN output, and comparing subsets of units by energy with a least-squares refit. I'll test sizes from 512 to 2048 and hold out half the tokens to check for overfitting, using ridge toward the original weights.



I'm checking the normalization of the exposed states, then ranking layer 0's FFN units by contribution energy and refitting the kept units' output weights with ridge, tested on held-out tokens.



## tool_use: run_python

```json
{
 "code": "exec(open('lib.py').read()); exec(open('lib2.py').read())\nimport scipy.special as sp\ndef gelu(x): return 0.5*x*(1+sp.erf(x/np.sqrt(2)))\ndef load_expo(k, l):\n    Z=np.load(f'pass_{k}/token_hidden_L{l+1}.npy' if False else f'pass_{k}/token_hidden_L{[int(f.split(\"L\")[1][:-4]) for f in os.listdir(f\"pass_{k}\") if f.startswith(\"token_hidden\")][0]}.npy')\n    m=np.load(f'pass_{k}/attention_mask.npy').astype(bool)\n    sent=np.repeat(np.arange(Z.shape[0])[:,None],Z.shape[1],1)[m]\n    return Z[m].astype(np.float64), sent\nimport os\nZ,sent=load_expo(10,0)\nprint(Z.shape, Z.mean(1)[:3], Z.std(1)[:3])\nP='bert.encoder.layer.0.'\ng1=base(P+'attention.output.LayerNorm.weight'); b1n=base(P+'attention.output.LayerNorm.bias')\nX=Z*g1+b1n\nW1=base(P+'intermediate.dense.weight').astype(np.float64); bb1=base(P+'intermediate.dense.bias')\nW2=base(P+'output.dense.weight').astype(np.float64); bb2=base(P+'output.dense.bias')\nA=gelu(X@W1.T+bb1); F=A@W2.T+bb2\nprint('|X| %.2f |F| %.2f |F-meanF| %.2f'%(np.linalg.norm(X,axis=1).mean(), np.linalg.norm(F,axis=1).mean(), np.linalg.norm(F-F.mean(0),axis=1).mean()))\ntr=sent%2==0; te=~tr\ndef prune_eval(A,F,W2,bb2,keep,lam):\n    # ridge toward original on tr, eval on te\n    As=A[:,keep]; mu=As[tr].mean(0); Fm=F[tr].mean(0)\n    Ac=As[tr]-mu; Fc=F[tr]-Fm\n    W0=W2[:,keep].T  # m x 768\n    Wn=np.linalg.solve(Ac.T@Ac+lam*np.eye(len(keep)), Ac.T@Fc+lam*W0)\n    b=Fm-mu@Wn\n    pred=As@Wn+b\n    err=np.linalg.norm(pred[te]-F[te])**2/np.linalg.norm(F[te]-F[te].mean(0))**2\n    # naive: original weights + mean comp\n    rem=np.setdiff1d(np.arange(A.shape[1]),keep)\n    pn=As@W0+bb2+ (A[tr][:,rem].mean(0))@W2[:,rem].T\n    errn=np.linalg.norm(pn[te]-F[te])**2/np.linalg.norm(F[te]-F[te].mean(0))**2\n    return err, errn\nen=A[tr].var(0)*(W2**2).sum(0)\nfor m in [512,1024,1536,2048]:\n    keep=np.sort(np.argsort(-en)[:m])\n    out=[prune_eval(A,F,W2,bb2,keep,lam) for lam in [1,10,100]]\n    print(m, 'LS(lam1,10,100) rel err:', ['%.3f'%o[0] for o in out], 'naive %.3f'%out[0][1])\n"
}
```

