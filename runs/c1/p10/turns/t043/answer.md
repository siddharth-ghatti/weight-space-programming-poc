## reasoning summary

I'm noting that lib.py is short and doesn't contain the LOO kernel ridge regression code, so I'll have to write that part myself. I'm sketching a closed-form LOO KRR built on a mix of a layer-normalized term and a second term, with normalization applied only over the non-reserved dimensions.

I'm checking the leave-one-out closed form for kernel ridge regression, using A = K + lam·I and the residual alpha_i / (A^-1)_ii, without an intercept. Then I'm matching hidden-state indices in pass_5: edited index 9 and base M9 both mean the output after original layer 8, and I'll compare them. Finally I'm running LOO directly on cls_hidden_L10, since that is what the pooler sees.



## message

lib.py only contains linear ridge, so I'll rewrite the closed-form LOO KRR. Then I'll compare pass_5 features with the base on the same 1024 ids.

## tool_use: run_python

```json
{
 "code": "import json\ncode = r'''\nimport numpy as np\nRES = np.array([83,88,222,254,444,495,506,601])\nKEEP = np.setdiff1d(np.arange(768), RES)\ndef lnr(X):\n    X = X[:, KEEP] if X.shape[1]==768 else X\n    X = X - X.mean(1, keepdims=True)\n    return X / np.sqrt((X**2).mean(1, keepdims=True) + 1e-12)\ndef zs(Xtr, Xte=None):\n    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-6\n    return ((Xtr-mu)/sd, None if Xte is None else (Xte-mu)/sd)\ndef rbf(A, B, gamma):\n    d = A.shape[1]\n    A2=(A**2).sum(1)[:,None]; B2=(B**2).sum(1)[None]\n    D = np.maximum(A2+B2-2*A@B.T, 0)/d\n    return np.exp(-gamma*D)\ndef krr_loo(X, y, gamma=0.4, lam=0.003, C=8):\n    Z,_ = zs(X)\n    K = rbf(Z, Z, gamma)\n    T = np.eye(C)[y]; T = T - T.mean(0)\n    Ainv = np.linalg.inv(K + lam*np.eye(len(y)))\n    al = Ainv@T\n    P = T - al/np.diag(Ainv)[:,None]\n    return (P.argmax(1)==y).mean()\n'''\nopen('krr.py','w').write(code)\nexec(code)\nd=np.load('base_feats.npz'); y=d['y']\nids=json.load(open('ids.json')); print(type(ids), len(ids))\nm5=json.load(open('pass_5/meta.json')); print(m5.keys()); \npr = {r['id']:r for r in practice()}\nids5 = m5.get('ids'); print(len(ids5), ids5[:3], ids[:3] if isinstance(ids,list) else list(ids.keys()))\n"
}
```

