## reasoning summary

The fitted output from Kc@A + b0 already tracks the expected logits closely (correlation around 0.997), with the remaining differences coming from the U projection and layer norm. Now I'm setting up the honest cross-validation: each fold trains on the practice data outside that fold plus my own examples at weight 0.3, using a dual weighted ridge. I think the solution has the form (K + lam W^-1) A = T - b0, but I still need to confirm that against the stated objective.

I'm now testing a guess at the solver form: kernel ridge with weights, where I'd check whether (Kc + lam*diag(1/w)) A matches T - b0, with T being either one-hot targets or one-hot minus 1/8.



## tool_use: run_python

```json
{
 "code": "\nimport json\nfrom scipy.special import erf\nk=np.load('kernel_v3.npz')\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\nC=k['C']; Kc=gelu(C@C.T/768); A=k['A']; yc=k['yc']; w=k['w']; lam=float(k['lam'])\nfor T in [np.eye(8)[yc], np.eye(8)[yc]-1/8]:\n    for name,M in [('K+lam/w',Kc+lam*np.diag(1/w)),('K+lam',Kc+lam*np.eye(len(w)))]:\n        r=M@A-(T-k['b0']); print(name, np.abs(r).max())\nprint(k['b0'], np.eye(8)[yc].mean(0))\n"
}
```

