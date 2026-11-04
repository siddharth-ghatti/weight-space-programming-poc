## reasoning summary

I want to check the hidden-state values before the second pass: the hidden state at position 9 should match the base M9, and the CLS vector at layer 10 should be roughly the layer norm of C9 plus 8 times M9. A quick numpy check on those read vectors should confirm it.



## message

Before spending the second read, a quick sanity check: does hidden 9 match the base reads, and does the final CLS look like LN(C9 + 8·M9)?

## tool_use: run_python

```json
{
 "code": "d=np.load('base_reads.npz'); R=np.load('reserved.npy')\nM9=np.load('pass_9/mean_hidden_L9.npy'); C9=np.load('pass_9/cls_hidden_L9.npy'); U=np.load('pass_9/cls_hidden_L10.npy')\nprint('M9 vs base', np.abs(M9-d['mean_hidden_L9'][:1024]).max(), 'C9 vs base', np.abs(C9-d['cls_hidden_L9'][:1024]).max())\nv=C9+8*M9; uh=(v-v.mean(1,keepdims=True))/v.std(1,keepdims=True)\nprint('U vs LN(C9+8M9) maxabs', np.abs(U-uh).max(), 'U mean/std', U.mean(1)[:3], U.std(1)[:3])\nprint('U reserved sample', U[0,R])"
}
```

