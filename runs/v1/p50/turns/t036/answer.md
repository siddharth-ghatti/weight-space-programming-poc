## reasoning summary

Accuracy is only 51%, far too low to be a small discrepancy, so something in the decoding must be off. I'm rechecking the scaling: the installed weights seem to be gw·W_l/ā_l, but the sum over the final values doesn't obviously match that.

The decode scaling checks out on paper: installed value is gw·W_l/ā_l, so decoding with 1/gw recovers the right sum. The 0.1 gain is consistent between measurement and install, so the scales themselves look correct. The new suspect is the decode weight of 1/gw = 10, which is about 4.6 times larger than before. Any common shift in the R coordinates now gets multiplied by 10 into every logit equally, and that could push tanh into saturation.

I'm going to decode the R coordinates with zero-sum weights, so each logit is 10 times the R coordinate minus the mean of all R coordinates. That removes the shared shift before it saturates tanh. The carried logits already sum to zero across classes, so centering the decode loses nothing. I'll also drop the sp scale to 0.5 to keep tanh roughly linear.

Centering the R block with zero-sum weights, 10·(I − 11ᵀ/8), should cancel the shared shift that was pushing tanh into saturation.



Accuracy is stuck at 51% because tanh saturates: the shared LayerNorm shift, amplified by decode weight 10, pushes all eight outputs toward ±1 so the argmax is noise. I'm switching the R block to a zero-sum decode, 10·(I − 11ᵀ/8), which removes that shift exactly without losing anything.



## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'build2.py').read())\nJ=np.load(WORK/'joint_v3.npz'); gw=float(J['gw']); sp=float(J['sp']); const=J['const']\nWfin=J['c12'].T.copy()\nCm=(np.eye(8)-np.ones((8,8))/8)/gw\nfor k in range(8):\n    for j in range(8): Wfin[k,R[j]]=Cm[k,j]\npw=np.zeros((768,768),np.float32); pb=np.zeros(768,np.float32); pw[:8]=sp*Wfin; pb[:8]=sp*const\nsave_edit('bert.pooler.dense.weight',pw); save_edit('bert.pooler.dense.bias',pb)\nprint('saved; check W(pooler) R block row0', W('bert.pooler.dense.weight')[0,R].round(3))\nprint('c12 block column sums (should be ~0):', np.abs(J['c12'].sum(1)).max())\n"
}
```

