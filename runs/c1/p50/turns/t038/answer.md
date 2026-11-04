## reasoning summary

I'm going to evaluate the 768-dimensional features as 12 blocks of 64, then sanity-check the uniform head block h10 against the base M0 projected onto Vdirs. After that I'll run kernel CV over M0 and the blocks, plus ablations. I also need to keep in mind that the class pools use W fit on all labels, which is a mild source of leakage.



## message

Evaluating E1. First a sanity check: does the uniform-head block match the base layer-0 mean projected onto the same value directions? Then kernel CV of M0 plus each pool group, as ablations.

## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\nfrom scipy.special import erf\ngelu=lambda t: 0.5*t*(1+erf(t/np.sqrt(2)))\nd=load_base(); y=labels()\nU=np.concatenate([np.load('pass_6/cls_hidden_L1.npy'),np.load('pass_7/cls_hidden_L1.npy')]); np.save('E1_u.npy',U)\npd=np.load('pool_design.npz'); W=pd['W']; pca=pd['pca']\nWn=W/np.linalg.norm(W,axis=0); Q,_=np.linalg.qr(np.concatenate([Wn.T,pca[:56]],0).T); Vd=Q.T\nM0=d['mean_hidden_L0']; ref=M0@Vd.T\nblk=lambda h: U[:,h*64:(h+1)*64]\ncc=[np.corrcoef(blk(10)[:,j],ref[:,j])[0,1] for j in range(8)]; print('uniform head vs M0 proj corr', np.round(cc,4))\ndef cosnorm(X):\n    Z=X-X.mean(0); Z=Z-Z.mean(1,keepdims=True); return Z/np.linalg.norm(Z,axis=1,keepdims=True)\ndef kern(Z,a=1.0): S=Z@Z.T; return S+(gelu(a*S)+gelu(-a*S))/(a*a)\ndef ev(bl, name, L=(0.01,0.03,0.1)):\n    Z=cosnorm(blockn(bl)); K=kern(Z)\n    print(name,'10f',krr_cv(K,y,L,k=10,seeds=(0,1,2)),'20f',krr_cv(K,y,L,k=20,seeds=(0,1)))\nev([M0],'M0')\nev([U],'U all')\nev([M0,U],'M0+U')\nev([M0]+[U[:,:512]],'M0+class pools')\nev([M0,U[:,512:640]],'M0+early pools')\nev([M0,U[:,704:768]],'M0+late pool')\nev([M0,U[:,512:640],U[:,704:]],'M0+pos pools')"
}
```

