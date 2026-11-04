## reasoning summary

I'm working out how the attention heads read and write: the query bias of 8 on dimension 0 and the key weights form a key direction, while the value and output projections look like a scaled permutation mapping the dimensions across. I'm also wondering whether the output bias is there to cancel the constant CLS residual after layer norm.

I'm checking whether the first-layer value dimensions match the classifier directions or the PCA components, starting with the first few value rows of each head against the pooling design weights. If the class directions are dropped from the output, I'd expect the cross-validation score to shift, though the attention weights still leak some signal.



E1's heads all attend by key direction, and each output dimension of U maps to one head and value row. Next I'm finding which value rows carry the label-fitted class directions, then rerunning CV with those dimensions removed.



## tool_use: run_python

```json
{
 "code": "exec(open('lib.py').read())\ny=labels()\nL='bert.encoder.layer.11.'\nvw=W(L+'attention.self.value.weight'); ow=W(L+'attention.output.dense.weight'); kw=W(L+'attention.self.key.weight')\npdz=np.load('pool_design.npz'); Wc=pdz['W']; pca=pdz['pca']\n# which value rows match class dirs\nfor h in [0,8]:\n    s=64*h\n    V=vw[s:s+64]\n    cw=np.abs((V/np.linalg.norm(V,axis=1,keepdims=True))@(Wc/np.linalg.norm(Wc,axis=0))).max(1)\n    cp=np.abs((V/np.linalg.norm(V,axis=1,keepdims=True))@(pca/np.linalg.norm(pca,axis=1,keepdims=True)).T).max(1)\n    print(h, np.round(cw[:12],2), np.round(cp[:12],2))\n# key row 0 vs Wc\nfor h in range(12):\n    k=kw[64*h]\n    c=(k/ (np.linalg.norm(k)+1e-9))@(Wc/np.linalg.norm(Wc,axis=0))\n    print(h, np.round(c,2), np.abs(kw[64*h+1:64*h+64]).max())\n# mapping of U dims\nperm=np.argmax(np.abs(ow),axis=0)  # col j (value dim) -> row (U dim)\nprint(perm[:10], perm[64:74])\nU=np.load('E1_u.npy').astype(np.float64)\ncls_dims=np.concatenate([perm[64*h:64*h+8] for h in range(12)])\nrest=np.setdiff1d(np.arange(768),cls_dims)\nprint('rest lin10', cv_acc(U[:,rest],y,lams=(0.1,1,10),k=10))\n"
}
```

