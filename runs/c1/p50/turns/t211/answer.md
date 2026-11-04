## reasoning summary

I'm checking pool_design.npz first, and it looks like it comes from the older E1 design (PCA and token-mean), so it isn't the current pool. The current pool layer was installed in run 5 for the 10-layer structure. With the query at zero, every score is zero whatever the key is, so attention comes out uniform and the key weight doesn't matter, even though its norm is 14.2.

I'm working out the attention side of the pool layer: with uniform attention over all non-padded tokens, including CLS and SEP, the value projection is the identity and the output dense is a diagonal. The mean of 19.8 with std 1.93 fits 760 dimensions at 20 and the 8 residual dimensions at about 1, which would mean c = 20 for the KEEP dimensions.

The residual dimensions are the ones where the scale is about 1 and the bias is roughly -mu9, so they just carry the raw mean plus CLS. I still need to check whether the installed c is 20 or the default 10, since the notes give 10 and the fitted diagonal suggests 20.

I'm checking whether the numpy prediction used the installed c, since the logits match to about 6e-6, so the verification pass probably did. I'm also working through the intermediate dense rows: the unit anchor directions scaled by 1/sqrt(768) give row norms that don't quite match the reported weight norm of 4.161.

I'm working out how the linear term reaches the class scores: the carried score goes into the RES dimensions of the attention output, and the FFN output gets added on top of it through the residual. After LayerNorm, the pooler reads those RES rows as the centered logits.

I'm checking how the linear term could be realized. The pool layer never sees M0, so a 760-dimensional linear feature can't be computed there, which makes me wonder how the joint fit was meant to be implemented.

I'm now checking the saved carry and LayerNorm design files for anything that records how the pool layer was installed, since the install scripts themselves aren't kept.

I'm now checking how the final scores come out of the pooler: it only reads the eight residual dimensions, so the classes depend on those rows after the final LayerNorm, which is set by the large key-independent attention term.

I'm now checking how the joint fit was meant to reach the class scores: whether the linear part uses the 760-dimensional M0 features or only the eight carried residual dimensions after the pool layer. The notes point to the full M0 linear features, which the pool layer never sees, so I still need to resolve that mismatch.

I'm checking whether the linear term could be realized as a pair of GELU units over the eight residual dimensions, so the pool layer could carry the class-score mapping through the residual path. If the joint fit used those eight carried values as its linear features, the final scores would be the carried term plus that mapping plus the kernel part.

I'm now checking whether the CV setup, which refits the linear part on 760 dimensions per fold, matches what was actually installed, since the installed linear weights came from a fit on all 1496 samples. I still need to find the install details in the carry provenance file.



The pool layer is now mapped: uniform attention, identity value, a diagonal output dense, and FFN rows that write only to the reserved dims. pool_design.npz is stale, so I'm checking carry_prov.npz and the 8-dim linear block next.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nimport numpy as np\nfrom poolfit import *\nd=np.load('carry_prov.npz'); print({k:d[k].shape for k in d.files})\nfor k in d.files:\n    if d[k].size<=16: print(k,d[k])\nprint(np.load('pool1_piv.npy').shape, np.load('pred_logits_pool1.npy').shape)\nP='bert.encoder.layer.11.'\nW1=W(P+'intermediate.dense.weight'); b1=W(P+'intermediate.dense.bias'); W2=W(P+'output.dense.weight')\nimport json; st=json.load(open('structure.json')); print(st.keys())\nkeep11=np.array(st['ffn_keep']['11'] if '11' in st['ffn_keep'] else st['ffn_keep'][str(11)])\nprint(len(keep11), keep11[:20], keep11[-20:])\nrn=np.linalg.norm(W1[keep11],axis=1); print('row norms', np.round(np.sort(rn)[:5],4), np.round(np.sort(rn)[-20:],3))\nbig=keep11[rn>0.05]; print('big units', big, rn[rn>0.05].round(3))\nprint('W1 big rows nnz', [(np.abs(W1[u])>1e-9).sum() for u in big], 'nnz cols at RES', [np.nonzero(np.abs(W1[u])>1e-9)[0][:8] for u in big[:3]])\nprint('W2 RES norms per unit (big)', np.linalg.norm(W2[RES][:,big],axis=0).round(3))\nprint('nonkept nonzero rows', (np.linalg.norm(np.delete(W1,keep11,0),axis=1)>0).sum())"
}
```

