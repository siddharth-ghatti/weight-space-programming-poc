## reasoning summary

I'm working out the analysis now: GELU in its exact form, 0.5x(1 + erf(x/√2)), using scipy's erf. For each layer I'm tracking the FFN input, the pre-activation and GELU output, the second projection, and the residual before LayerNorm 2, and I'm starting to set up how the relative error from pruning units would be measured.

I'm weighing how to measure the pruning error: since LayerNorm 2 removes the per-token mean and scales by the std, I'm leaning toward a per-token relative error, the norm of the removed contribution over the norm of the centered residual. I'm also planning three selection scores to compare: energy, variance (with the mean folded into the bias), and least-squares compensation chosen by greedy or pivoted QR. The check will run on layers 0-4 across 5049 tokens, split by paragraph to test whether the compensation generalizes, and on layer 5 using 1000 CLS tokens.



Next I'll run the offline FFN analysis, measuring pruning error on held-out paragraphs to compare energy and variance ranking rules across several widths.



## tool_use: run_python

```json
{
 "code": "from scipy.special import erf\nimport json\ndef gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))\n# paragraph index per token for split\nM = np.load(WORK/'pass_4/attention_mask.npy').astype(bool)\npara = np.repeat(np.arange(32)[:,None], M.shape[1], 1)[M]\ntr = para%4!=0; te = ~tr\nres = {}\nfor Lr in range(6):\n    pre = f\"roberta.encoder.layer.{Lr}.\"\n    W1 = base(pre+\"intermediate.dense.weight\"); b1 = base(pre+\"intermediate.dense.bias\")\n    W2 = base(pre+\"output.dense.weight\"); b2 = base(pre+\"output.dense.bias\")\n    if Lr<5:\n        A = np.load(WORK/f'ffn_in_L{Lr}.npy'); trm, tem = tr, te\n    else:\n        A = np.load(WORK/'ffn_in_L5_cls.npy'); idx = np.arange(len(A)); trm = idx%4!=0; tem = ~trm\n    H = gelu(A@W1.T + b1)\n    Out = H@W2.T + b2; R = A + Out\n    Rc = R - R.mean(1, keepdims=True)\n    wn2 = (W2**2).sum(0)\n    energy = (H[trm]**2).mean(0)*wn2\n    var = H[trm].var(0)*wn2\n    mu = H[trm].mean(0)\n    line=[]\n    for m in [256,512,768,1024,1536,2048]:\n        for nm,sc in [('E',energy),('V',var)]:\n            keep = np.argsort(-sc)[:m]; drop = np.setdiff1d(np.arange(3072), keep)\n            D = (H[tem][:,drop] - mu[drop])@W2[:,drop].T   # error after mean-compensation\n            Dc = D - D.mean(1, keepdims=True)\n            rel = np.linalg.norm(Dc,axis=1)/np.linalg.norm(Rc[tem],axis=1)\n            line.append(f\"{nm}{m}:{rel.mean():.3f}\")\n    # fraction of FFN output relative to residual\n    Oc = Out - Out.mean(1,keepdims=True)\n    print(f\"L{Lr} ffn/resid {np.mean(np.linalg.norm(Oc[tem],axis=1)/np.linalg.norm(Rc[tem],axis=1)):.3f} | \"+' '.join(line))"
}
```

