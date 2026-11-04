## reasoning summary

Layer 1 of the attention output maps to encoder layer 0, since layers 1 through 12 correspond to encoders 0 through 11. For each layer and head, I'll average the attention that non-CLS, non-pad queries put on the CLS key, and also check attention to SEP.



## message

Pass 6 has the attention maps (attention_L1 = encoder layer 0). Next I'll compute how much attention mass non-[CLS] tokens put on [CLS], per layer and head.

## tool_use: run_python

```json
{
 "code": "am=np.load('pass_4/attention_mask.npy').astype(bool)\nres=[]\nfor L in range(1,13):\n    A=np.load(f'pass_6/attention_L{L}.npy')  # n,h,T,T\n    n=A.shape[0]\n    tocls=[]; tosep=[]\n    for i in range(n):\n        Ti=am[i].sum()\n        a=A[i,:,1:Ti,:Ti]  # queries non-cls\n        tocls.append(a[:,:,0].mean(1)); tosep.append(a[:,:,Ti-1].mean(1))\n    tocls=np.array(tocls).mean(0); tosep=np.array(tosep).mean(0)\n    print(f'enc layer {L-1}: toCLS mean {tocls.mean():.3f} max-head {tocls.max():.3f} | per head', np.round(tocls,2), '| toSEP mean', round(tosep.mean(),3))"
}
```

