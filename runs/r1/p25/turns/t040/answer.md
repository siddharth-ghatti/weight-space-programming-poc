## reasoning summary

I'm checking whether the CLS token separates linearly from other tokens across layers 0 to 3, which it does with a wide margin early on and a narrower one at layer 3. Layer 4 is borderline, so a better direction like LDA might help. The </s> token separates in layers 0 to 3 with smaller margins, but not in layer 4 with this simple projection. I'm also wondering whether the max-other estimate, based on only about 4000 tokens from 32 paragraphs, will hold up on other texts.

I'm picking the threshold midway between the non-CLS max and the CLS min, then scaling so CLS lands near +8 and every other token falls to roughly -7.7, where GELU is effectively zero. Next I'm checking whether LDA gives a bigger margin in standard-deviation units than the mean-difference direction, and then working out how much CLS error the pruning actually introduces.

I'm now comparing the size of the pruning error vector against the pre-LN2 magnitude and the CLS output spread, and I'm also weighing whether a CLS-specific linear correction could beat a single detector. For the </s> token, I'll apply a similar mean correction using its own detector in layers 0 to 3, where its within-token spread looks noticeably larger.

Next I'm checking the error for the </s> token as well, and I'm considering whether a detector unit needs its own slot, which could mean giving up one or two of the linear unit slots. After that I'm computing the error norms for CLS and </s> at the current width of 1536, and comparing them to CLS's pre-LN magnitude.



CLS separates cleanly from other tokens through layer 2 and `</s>` through layer 3, so I'm now estimating the mean FFN-output error at both tokens against their token-level spread.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK)); import ffnlib as F; from lu import *\nkind=np.load('tok_kind.npy')\ndef ffn_err_special(l, m, k):\n    e,r,keep,pr,M,Ur,sv=plan_layer(l,m,base)\n    d=np.load(f'timp_L{l}.npz'); Egt,mut=d['Eg'],d['mu']\n    a=np.load(f'true_a_L{l}.npy').astype(np.float64); ak=a[kind==k]\n    muk=ak.mean(0); covk=np.cov(ak.T)\n    W1=base(f'roberta.encoder.layer.{l}.intermediate.dense.weight').astype(np.float64); b1=base(f'roberta.encoder.layer.{l}.intermediate.dense.bias').astype(np.float64)\n    U=base(f'roberta.encoder.layer.{l}.output.dense.weight').astype(np.float64)\n    mk,sk=F.unit_stats(W1,b1,muk,covk); Egk,Vgk=F.gauss_gelu_moments(mk,sk)\n    Mr=Ur@(Ur.T@M)\n    err=U[:,pr]@(Egk[pr]-Egt[pr]) - Mr@(muk-mut)\n    full_out=U@Egk\n    return np.linalg.norm(err), np.linalg.norm(full_out), np.sqrt((Vgk*(U**2).sum(0)).sum())\nfor l in range(5):\n    for k,name in [(0,'CLS'),(1,'</s>')]:\n        en,fn,sp=ffn_err_special(l,1536,k)\n        print(f'L{l} {name}: mean-error norm {en:.2f}, full FFN out norm {fn:.2f}, approx within-{name} FFN spread {sp:.2f}')\n    a=np.load(f'true_a_L{l}.npy').astype(np.float64); at=a[kind==2]; ct=np.cov(at.T)+1e-3*np.eye(768)\n    for k,name in [(0,'CLS'),(1,'</s>')]:\n        dd=np.linalg.solve(ct,a[kind==k].mean(0)-at.mean(0)); p=a@dd; sd=p[kind==2].std()\n        print(f'   LDA {name}: other max {(p[kind==2].max()-p[kind==2].mean())/sd:.1f} sd, {name} min {(p[kind==k].min()-p[kind==2].mean())/sd:.1f} sd')"
}
```

