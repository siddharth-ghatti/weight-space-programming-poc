## reasoning summary

I'm reconstructing the kernel CV by rebuilding Z from the attention output: with v2 the attention weights are uniform and O=2I, so the output is 2*mean11 added to the residual cls11, then layer-normalized with gamma1 and beta0. I'm about to work out what the FFN input becomes from there.

Checking the attention-to-FFN relation: the pre-activation works out to beta times the dot product of Z_i and Z, divided by 768. I'm also verifying that mean11 covers all tokens including CLS and SEP, which the earlier v1 check supports. Next I'll write the CV kernel and list the errors.



## message

Next I'll rebuild the kernel features (as in v3) and run honest CV to list which practice sentences get misclassified.

## tool_use: run_python

```json
{
 "code": "\nk=np.load('kernel_v3.npz'); print({n:k[n].shape for n in k.files})\nfor n in k.files:\n    if k[n].size<10: print(n,k[n])\nprint(np.load('folds.npy').shape, np.load('folds.npy').max())\nm=json.load(open('my_sents_all.json')); print(len(m), m[0])\nprint(os.listdir('pass_10'))\n"
}
```

