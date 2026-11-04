## message

Before spending the holdout pass, a quick check on these 1024 that v2's carried signal now matches its proxy M1@A better than v1's did.

## tool_use: run_python

```json
{
 "code": "exec(open('helpers.py').read())\nlab=labels_all(); y=lab[:1024]\nD0=np.load('D_dims.npy'); A=np.load('A_proj_v2.npy'); Cp=np.eye(8)-1/8\nM1 = load('mean',1,'pass_3').copy(); M1[:,D0]=0\nc12 = load('cls',12,'pass_5'); m11=load('mean',11,'pass_5')\nact = m11[:,D0]@Cp; prox=(M1@A)@Cp\nprint('v2 corr actual(L11) vs proxy', np.round([np.corrcoef(act[:,j],prox[:,j])[0,1] for j in range(8)],3))\nprint('D contrast rms L1 %.4f L11 %.4f'%(np.sqrt(((load('mean',1,'pass_5')[:,D0]@Cp)**2).mean()), np.sqrt((act**2).mean())))\nfolds=make_folds(1024,5,0)\nprint('all', ridge_cv(c12,y,(0.1,0.2,0.4),folds))\n"
}
```

