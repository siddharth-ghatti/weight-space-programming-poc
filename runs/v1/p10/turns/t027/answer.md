## reasoning summary

I'm noting that the correlation sits between 0.57 and 0.9, so the carried signal departs from the proxy quite a bit. Three candidate causes stand out: the token weighting κ_t, applying A to the post-attention x0 rather than the embedding e, and the final layer normalization that scales by the sentence-level std, which hits every dimension including D. The 1_D offset is also stripped out by Cp.

I'm checking whether the final layer-norm scale is the main reason the carried signal departs from the proxy, by comparing the harness's mean hidden state against c12 and seeing whether they match.

I'm weighing whether the proxy-based expectation is a fair baseline: if fitting on m0 through A would reach about 80% on the 472 holdout, then the 71% actual result means the lost fidelity is costly and better matching the carried signal would help. I'll compute that proxy D-only holdout score to find out.



The real carried signal only correlates 0.57–0.90 with its proxy, so fidelity is being lost somewhere. I'm checking whether the final LayerNorm or the per-token rescaling is the cause.



## tool_use: run_python

```json
{
 "code": "exec(open('helpers.py').read())\nlab=labels_all()\nD = np.load('D_dims.npy'); A=np.load('A_proj.npy')\nc12 = np.concatenate([load('cls',12,'pass_3'), load('cls',12,'pass_4')])\nm11 = np.concatenate([load('mean',11,'pass_3'), load('mean',11,'pass_4')])\nm0 = np.concatenate([load('mean',0,'pass_3'), load('mean',0,'pass_4')])\ndef LN(X): X=X-X.mean(1,keepdims=True); return X/np.sqrt((X**2).mean(1,keepdims=True))\nprint('max |c12 - LN(m11)|', np.abs(c12-LN(m11)).max(), 'typical', np.abs(c12).mean())\nCp=np.eye(8)-1/8\nact = m11[:,D]@Cp; prox=(m0@A)@Cp\nprint('pre-LN corr', [round(np.corrcoef(act[:,j],prox[:,j])[0,1],3) for j in range(8)])\nsc = np.sqrt(((m11-m11.mean(1,keepdims=True))**2).mean(1)); print('final LN scale std/mean', sc.std()/sc.mean())\nytr,yte=lab[:1024],lab[1024:]\nfor name,X in [('proxy',m0@A),('actual pre-LN',m11[:,D]),('actual c12',c12[:,D])]:\n    Wf,b=ridge_fit(X[:1024],ytr,0.01); p=(X[1024:]@Wf+b).argmax(1); print(name,'D-only holdout acc %.3f'%(p==yte).mean())\n"
}
```

