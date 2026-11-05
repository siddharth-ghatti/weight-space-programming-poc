## reasoning summary

I'm estimating how much the FFN output error at m=1536 (about 5-7% per layer) shifts the agreement figure, and I'm reasoning that perturbation size grows with the square root of the error, while disagreement grows roughly linearly with that perturbation.

I'm checking whether one layer is responsible for most of the error rather than the errors compounding evenly, and I'm weighing whether refitting the head would beat simply reducing the layer width.

I'm weighing whether a head refit or a correction to the final layer would do more for agreement, since the head can only see the last layer's CLS output. I'm planning to measure how much the CLS hidden state drifts layer by layer between pass 11 and pass 1 to locate where the disagreement comes from.



Linear-unit compensation roughly halves the error, though I estimate about 5% disagreement would remain even at m=2048 unless the error is concentrated somewhere. Next I'm tracing how the deviation from base builds up layer by layer in pass 11.



## tool_use: run_python

```json
{
 "code": "import numpy as np\ndef r2(a,b):\n    return 1-((a-b)**2).sum()/((b-b.mean(0))**2).sum()\nfor pd in ['pass_3','pass_11']:\n    print(pd)\n    for k in range(7):\n        cb=np.load(f'pass_1/cls_hidden_L{k}.npy'); ce=np.load(f'{pd}/cls_hidden_L{k}.npy')\n        mb=np.load(f'pass_1/mean_hidden_L{k}.npy'); me=np.load(f'{pd}/mean_hidden_L{k}.npy')\n        print(f'  L{k}: cls R2 {r2(ce,cb):.4f} relerr {np.linalg.norm(ce-cb)/np.linalg.norm(cb-cb.mean(0)):.3f} | mean R2 {r2(me,mb):.4f}')\n"
}
```

