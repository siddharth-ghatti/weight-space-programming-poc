## reasoning summary

I'm checking the layer-12 value against the mean of the hidden states at layer 11, and the gap is only about 0.01, so the final layer norm isn't what's breaking fidelity. The drop has to come from the κ_t weighting or from using x0 instead of e. The proxy-only setup scores 88% on the holdout against 72% for the real model, so there's room to gain by tightening that fidelity, and the combined stage-2 accuracy with the proxy was already around 97.5%.

I'm now checking whether the κ_t weighting or the x0-versus-e choice is the bigger source of loss. The layer-1 signal is just one layer-norm factor away from the input, so I'll compare it against the proxy there and then track how the correlation drops off through layers 2 to 11.



The final [CLS] matches LayerNorm of the token mean, so the last norm isn't the culprit. The proxy hits 88% on the holdout while the real carried signal reaches only 72%, so I'm checking layer by layer where the correlation drops.



## tool_use: run_python

```json
{
 "code": "exec(open('helpers.py').read())\nD = np.load('D_dims.npy'); A=np.load('A_proj.npy')\nm0 = load('mean',0,'pass_3'); prox=(m0@A)@(np.eye(8)-1/8)\nCp=np.eye(8)-1/8\nfor l in range(1,12):\n    act = load('mean',l,'pass_3')[:,D]@Cp\n    cs = [np.corrcoef(act[:,j],prox[:,j])[0,1] for j in range(8)]\n    print(l, 'mean corr %.3f'%np.mean(cs), np.round(cs,2))\n"
}
```

