## reasoning summary

The shallow model seems to miss negation-based semantic errors, like "not insured" versus coverage, because the kernel on mean pooling can't capture the conjunction well. Deeper layers should handle that better, so I'm leaning toward needing depth, and the final size lands around 95.3M with k=10 and m=3072 unless I can find savings. Wait, maybe I should reconsider that.

I'm weighing whether a moderate width cut, something like m=2048 or 2560 with the readout refit on the real features, could hold up where m=1024 failed. The compensation step may be the real problem, since dropped units leave a systematic shift that grows with depth. Units that are almost always inactive could be removed at little cost, but the uniform width means the least-used layer sets the limit, so I'm checking how many units are near-dead.



The shallow model's extra errors come from negation and scope cases, so depth is needed. Next I'll count the nearly-always-inactive FFN units in each layer for mild pruning.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nfrom scipy.special import erf\nfor l in range(9):\n    W1=base(f'bert.encoder.layer.{l}.intermediate.dense.weight'); b1=base(f'bert.encoder.layer.{l}.intermediate.dense.bias')\n    g=base(f'bert.encoder.layer.{l}.attention.output.LayerNorm.weight'); be=base(f'bert.encoder.layer.{l}.attention.output.LayerNorm.bias')\n    mu=W1@be+b1; sd=np.sqrt((W1**2)@(g**2)); zt=mu/sd\n    print(l, 'units with mu+2sd<0:', int((mu+2*sd<0).sum()), ' mu+3sd<0:', int((mu+3*sd<0).sum()), ' mu+1.5sd<0:',int((mu+1.5*sd<0).sum()))\n"
}
```

