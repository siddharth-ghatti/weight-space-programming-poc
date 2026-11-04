## reasoning summary

The extra carry block would add roughly 0.3-0.5% on the 1024-item nested CV, which is about four items and within noise. Each extra block also protects eight more dimensions and adds more perturbation, so I'm leaning toward keeping it simple with layer 0 carry and refitting A on the 1496 set. I'm still weighing whether the (0,6) block is worth including.

I'm weighing whether to add a layer-6 carry block as well: the nested CV gain is small, about the same as before, and it would protect 16 dims instead of 8. The catch is that fitting A on all 1496 leaves no honest holdout, so I'd only be able to check it in-sample, which makes the extra block harder to validate.

I'm checking the layer-6 case more closely: the D6 dims already carry the original network's content before layer 6, so the write would add onto that, and I'm considering zeroing them at layer 6's LN first, the way layer 0 is handled. I also need to keep track of which hidden state the layer-6 FFN actually reads, since it's the LN output rather than the raw residual.

I'm weighing whether the layer-6 block is worth including, since the gain is about the same size as before and the validation would only be in-sample. The layer-6 FFN reads the LN output rather than the raw residual, so I'd need to handle the D6 dims carefully, and I'm going to decide whether to hold out data for v2 or accept the in-sample check.

I'm going to check whether the carried D-dimension signal in v1 tracks the proxy I used for the nested CV, by correlating the actual contrast values against m0@A across the 1024 and 472 items. If r comes out above 0.9, the proxy-based estimate is a fair guide for v2 even though A is fit in-sample there.



A second carried block adds about 0.4% in nested CV, within noise, so I'm checking how well its proxy m0@A tracks the real signal in v1 before relying on proxy-based CV for v2.



## tool_use: run_python

```json
{
 "code": "exec(open('helpers.py').read())\nD = np.load('D_dims.npy'); A=np.load('A_proj.npy')\nc12 = np.concatenate([load('cls',12,'pass_3'), load('cls',12,'pass_4')])\nm0 = np.concatenate([load('mean',0,'pass_3'), load('mean',0,'pass_4')])\nCp = np.eye(8)-1/8\nact = c12[:,D]@Cp; prox = (m0@A)@Cp\nfor j in range(8):\n    print(j, 'corr %.3f'%np.corrcoef(act[:,j],prox[:,j])[0,1])\n# best linear map actual <- proxy\nWl,res,_,_ = np.linalg.lstsq(np.c_[prox,np.ones(len(prox))], act, rcond=None)\npred = np.c_[prox,np.ones(len(prox))]@Wl\nprint('R2', 1-((act-pred)**2).sum(0)/((act-act.mean(0))**2).sum(0))\n"
}
```

