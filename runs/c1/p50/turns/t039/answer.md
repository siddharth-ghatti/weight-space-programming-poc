## reasoning summary

I'm comparing M0 at 97.68 with 10 folds and 97.9 with 20, which differ from the earlier 97.55 and 97.7 because this time I used all 768 dimensions instead of the 760 kept ones. Adding pools seems to hurt, since equal block weighting lets the many small pool blocks dilute M0, so I'm considering weighting the blocks differently.

I'm weighing the pool blocks down with a factor w, so the M0 features aren't drowned out. The uniform-head correlation with the M0 projection sits around 0.93 to 0.98, which is close enough given the layer-norm centering. Next I'll sweep w over 0.1, 0.2, 0.3 and 0.5.



## message

Adding the pools *hurt*. Normalizing every block to equal weight lets the eleven 64-dim pool blocks drown out the main mean. Next I'll down-weight the pool blocks relative to M0 and scan the weight.

## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\nfrom scipy.special import erf\ngelu=lambda t: 0.5*t*(1+erf(t/np.sqrt(2)))\nd=load_base(); y=labels(); U=np.load('E1_u.npy'); M0=d['mean_hidden_L0']\ndef cosnorm(X):\n    Z=X-X.mean(0); Z=Z-Z.mean(1,keepdims=True); return Z/np.linalg.norm(Z,axis=1,keepdims=True)\ndef kern(Z,a=1.0): S=Z@Z.T; return S+(gelu(a*S)+gelu(-a*S))/(a*a)\ndef nb(x): x=x-x.mean(0); return x/np.sqrt((x**2).sum(1).mean())\ndef ev(parts, name, L=(0.03,)):\n    Z=cosnorm(np.concatenate(parts,1)); K=kern(Z)\n    return krr_cv(K,y,L,k=10,seeds=(0,1,2))[0.03], krr_cv(K,y,L,k=20,seeds=(0,1))[0.03]\nbase_=ev([nb(M0)],'M0'); print('M0',base_)\ngroups={'class':U[:,:512],'early':U[:,512:640],'unif':U[:,640:704],'late':U[:,704:]}\nfor g,X in groups.items():\n    for w in [0.15,0.3,0.5]:\n        print(g,w,ev([nb(M0),w*nb(X)],g))"
}
```

