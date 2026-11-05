## reasoning summary

The residual-like behavior of layers 1 through 3 stands out, so dropping one of those might do the least damage. Layer 5 also shifts the CLS representation sharply, probably because of fine-tuning.

Next I'm working out the FFN importance per layer, starting with a helper module (ffn_imp.py) that later runs can import. For the GELU moments under a Gaussian input, I'll use Gauss-Hermite quadrature with 64 points, vectorized over the 3072 units, and I'll average importance across the layer input and the next layer's input.



Each layer shifts the residual stream a lot: an identity map explains only about 45–49% of token variance in the middle layers, and a diagonal affine map about 66%. Next I'll score FFN units by variance-weighted output norm and derive bias compensation for pruned ones.



## tool_use: run_python

```json
{
 "code": "code = r'''\nimport numpy as np\n_gh_x,_gh_w=np.polynomial.hermite_e.hermegauss(80)  # probabilists' Hermite: weight exp(-x^2/2)\n_gh_w=_gh_w/_gh_w.sum()\nfrom math import erf\nimport scipy.special as sps\ndef gelu(z): return 0.5*z*(1+sps.erf(z/np.sqrt(2)))\ndef gauss_gelu_moments(m,s):\n    z=m[:,None]+s[:,None]*_gh_x[None,:]\n    g=gelu(z)\n    Eg=(g*_gh_w).sum(1); Eg2=(g*g*_gh_w).sum(1)\n    return Eg, np.maximum(Eg2-Eg**2,0)\ndef unit_stats(W1,b1,mu,cov):\n    m=W1@mu+b1\n    s=np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',W1,cov,W1),1e-12))\n    return m,s\n'''\nopen(WORK/'ffnlib.py','w').write(code)\nimport sys; sys.path.insert(0,str(WORK))\nfrom ffnlib import *\nmus=np.load('tok_mu.npy'); covs=np.load('tok_cov.npy')\nimp={}\nfor l in range(6):\n    W1=base(f'roberta.encoder.layer.{l}.intermediate.dense.weight').astype(np.float64)\n    b1=base(f'roberta.encoder.layer.{l}.intermediate.dense.bias').astype(np.float64)\n    U=base(f'roberta.encoder.layer.{l}.output.dense.weight').astype(np.float64)\n    un=(U**2).sum(0)\n    res=[]\n    for src in [l,l+1]:\n        m,s=unit_stats(W1,b1,mus[src],covs[src])\n        Eg,Vg=gauss_gelu_moments(m,s)\n        res.append((m,s,Eg,Vg*un))\n    I_in,I_out=res[0][3],res[1][3]\n    rk_in=np.argsort(-I_in); rk_out=np.argsort(-I_out)\n    ov=[len(set(rk_in[:k])&set(rk_out[:k]))/k for k in [512,1024,1536]]\n    I=np.sqrt(I_in*I_out)\n    srt=np.sort(I)[::-1]; cum=np.cumsum(srt)/srt.sum()\n    wimp=np.linalg.norm(W1,axis=1)*np.sqrt(un)\n    ovw=len(set(np.argsort(-wimp)[:1024])&set(np.argsort(-I)[:1024]))/1024\n    print(f'L{l}: top-k overlap in/out proxies {np.round(ov,2)}, weight-vs-stat overlap@1024 {ovw:.2f}; retained var frac @512 {cum[511]:.3f} @768 {cum[767]:.3f} @1024 {cum[1023]:.3f} @1536 {cum[1535]:.3f} @2048 {cum[2047]:.3f}')\n    np.savez(f'imp_L{l}.npz',I=I,I_in=I_in,I_out=I_out,Eg_in=res[0][2],Eg_out=res[1][2],m_in=res[0][0],s_in=res[0][1],m_out=res[1][0],s_out=res[1][1])"
}
```

