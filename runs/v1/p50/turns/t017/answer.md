## reasoning summary

I'm checking the CLS and mean representations layer by layer against the scale s, starting from the base model's first 256 samples. Layers 0 and 1 should be unchanged, and the R coordinates at layer 2 should be zero, while layer 3 should carry the signal from layer 2 scaled by the factors. For each layer I'll fit R_k as a linear function of s plus a constant, then compare the slope's mean and spread across sentences along with the correlation and argmax agreement.

I'm also comparing the non-R coordinates at layer 2 against the base model, where they should match exactly, and then checking how much drift appears at layer 11.



I'm regressing each layer's R coordinates on the carried logits and checking argmax agreement, plus how far the non-R coordinates drift from the base model.



## tool_use: run_python

```json
{
 "code": "M=np.load(WORK/'base_mean.npy')[:,:256]; C=np.load(WORK/'base_cls.npy')[:,:256]\nD=np.load(WORK/'D.npy'); b0=np.load(WORK/'b0.npy'); lab=np.load(WORK/'labels.npy')[:256]\nR = [601, 222, 88, 83, 444, 506, 254, 495]; keep=np.setdiff1d(np.arange(768),R)\ns=M[2]@D.T+b0\nsc=s-s.mean(1,keepdims=True)\nfor k in range(13):\n    ce=np.load(WORK/f'pass_3/cls_hidden_L{k}.npy'); me=np.load(WORK/f'pass_3/mean_hidden_L{k}.npy')\n    out=[]\n    for name,arr in [('cls',ce),('mean',me)]:\n        r=arr[:,R]; rc=r-r.mean(1,keepdims=True)\n        a=(rc*sc).sum(1)/((sc**2).sum(1)+1e-9)   # per-sentence scale\n        resid=np.linalg.norm(rc-a[:,None]*sc,axis=1)/(np.linalg.norm(rc,axis=1)+1e-9)\n        agree=(r.argmax(1)==s.argmax(1)).mean()\n        out.append(f\"{name}: a={a.mean():.3f}\u00b1{a.std():.3f} relres={resid.mean():.3f} agree={agree:.3f}\")\n    dc=np.linalg.norm(ce[:,keep]-C[k][:,keep],axis=1).mean()/np.linalg.norm(C[k][:,keep],axis=1).mean()\n    dm=np.linalg.norm(me[:,keep]-M[k][:,keep],axis=1).mean()/np.linalg.norm(M[k][:,keep],axis=1).mean()\n    print(k, ' | '.join(out), f'| drift cls {dc:.3f} mean {dm:.3f}')\n"
}
```

