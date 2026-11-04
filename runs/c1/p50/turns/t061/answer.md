## reasoning summary

I'm planning the evaluation: 10-fold kernel cross-validation on the 1024 ids, possibly over several seeds, for D1 at each depth, compared against the base matching depth. I'll include M0+M as well.



## message

Now I'll compare D1 with base on the same 1024 ids, depth by depth (kernel CV, M alone and M0+M, averaged over 2 fold seeds).

## tool_use: run_python

```json
{
 "code": "exec(open('lib.py').read())\ny=labels()[:1024]; B=load_base(); D=loadpass(9)\ndef bn(X): X=X[:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.sqrt((X**2).sum(1).mean())\ndef kcv(Z, gs=(0.5,1), lams=(0.003,0.01,0.03), k=10):\n    Z=Z/np.linalg.norm(Z,axis=1,keepdims=True); S=Z@Z.T\n    return max(max(krr_cv(np.exp(g*(S-1)),y,lams=lams,k=k,seeds=(0,1)).values()) for g in gs)\nM0=bn(B['mean_hidden_L0'][:1024])\nprint('base:')\nbres={}\nfor l in [6,8,9,10,11]:\n    b=bn(B[f'mean_hidden_L{l}'][:1024]); bres[l]=(kcv(b),kcv(np.concatenate([M0,b],1)))\n    print(' L%d alone %.4f  +M0 %.4f'%(l,*bres[l]))\nmp={5:6,6:8,7:9,8:10,9:11}\nprint('D1 (drop 4,6):')\nfor e,bl in mp.items():\n    d=bn(D[f'mean_hidden_L{e}'])\n    print(' edited L%d (after orig %d): alone %.4f  +M0 %.4f'%(e,bl-1,kcv(d),kcv(np.concatenate([M0,d],1))))\n"
}
```

