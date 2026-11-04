## reasoning summary

The fit with per-sample k noise matches the joint fit at 98.04, so switching the carry direction to the joint-fit linear block and deploying that fit as-is, without refitting B, should bring back the cross-validated performance. I'm now checking whether the per-sample scale k_i behaves the same for the carried scores, since it comes from the LayerNorm scale factors along the residual dimensions.

I'm checking whether the LayerNorm scale factors k_i stay the same across the carried scores, since they depend on per-token std rather than the carry direction. Next I'm making sure the centering across the 8 residual dimensions removes any common offset before the deployed pool is applied.

I'm checking how the carried scores relate to the final residual output: the LayerNorm scaling and centering should pass the joint-fit linear block through without changing the argmax. I'm working out a carry scale λ so that the residual dimensions come out proportional to the joint-fit scores plus the deployed bias.

I'm working out how the total linear coefficient on the residual dimensions has to balance the carry scale and the deployed linear block, so that the combined output comes out as a clean multiple of the scores plus bias. I'm also checking whether the overall scale γ keeps the tanh-style normalization from saturating, given that the residual contribution should stay small compared with the sqrt(768) normalization term.

I'm checking whether the scale γ can stay near 1 so the residual contribution stays small against the normalization, and I'm noting that the per-sample variation in std from the added term is under 0.2%, which leaves argmax unaffected.

I'm checking that the GELU pair trick gives an exact linear pass-through, so the residual-stream units can carry z[RES] directly. I'm also noting that the carry's magnitude feeds into the LayerNorm normalization of z, so a changed carry scale shifts the KEEP-part scale slightly.

I'm checking whether the carry's magnitude matters here: the installed scale is c=20 while the fit used c=10, and since the KEEP part dominates the LayerNorm norm, the RES dimensions should barely shift it. The open question is that the measured features include the old carry in those dimensions, so the new carry would change x[RES]; I'm considering either matching the old magnitude or substituting the predicted new RES values into the numpy model, possibly by zeroing the RES dims in the kernel features.

I'm weighing whether the carry scale should be c=10 as in the CV runs or c=20 as installed, since the RES dimensions barely shift the LayerNorm normalization either way. Checking the RES energy against the total in x will show how much that matters, and I'm comparing the CV results at each setting to decide which to use.

I'm settling on c=10, λ=0.05, and c0=0.5 for the carry, since that combination held up slightly better in the cross-validation runs. Next I'm turning to the L0 head 0 weights, checking which query, value, and output entries changed and how the carry dimensions sit inside them.

I'm going to inspect the L0 head 0 value and output weights next, to see how the carry is laid out there, and I'll write the new carry direction in the same way so its magnitude matches the old one.

I'm now pulling up the L0 head 0 weights to see how the query, value, and output entries changed, and where the carry dimensions sit within them.



The joint fit reaches the full CV score even with the 15% scale noise, so I'll swap L0's carry direction for its linear block and then inspect how the carry is wired in head 0 and the output dense.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nimport numpy as np\nfrom poolfit import *\ncp=np.load('carry_prov.npz'); wraw=cp['wraw'].astype(np.float64); mu0=cp['mu0'].astype(np.float64); s=float(cp['s']); sc=float(cp['sc']); wlin=cp['wlin']\nP='bert.encoder.layer.0.'\nWv=W(P+'attention.self.value.weight').astype(np.float64); bv=W(P+'attention.self.value.bias').astype(np.float64)\nWo=W(P+'attention.output.dense.weight').astype(np.float64); bo=W(P+'attention.output.dense.bias').astype(np.float64)\nVc=Wv[56:64]; print('Vc shape', Vc.shape, 'RES cols of Vc nonzero', np.abs(Vc[:,RES]).max())\n# relation to wraw\nfor nm,cand in [('s*wraw',s*wraw.T),('sc*wraw',sc*wraw.T),('wraw',wraw.T)]:\n    print(nm, np.abs(Vc-cand).max(), np.linalg.norm(Vc)/np.linalg.norm(cand))\nprint('bv[56:64]', bv[56:64].round(4), ' -s*wraw^T mu0', (-s*wraw.T@mu0).round(4))\nprint('Wo[RES,56:64]', np.round(Wo[np.ix_(RES,range(56,64))],4))\nprint('Wo[KEEP,56:64] max', np.abs(Wo[np.ix_(KEEP,range(56,64))]).max(), 'Wo[RES, other] max', np.abs(Wo[RES][:, 64:]).max(), 'Wo[RES,0:56] max', np.abs(Wo[RES][:, :56]).max())\nprint('bo[RES]', bo[RES].round(4))\nprint('wraw RES rows', np.abs(wraw[RES]).max(), 'wraw norm', np.linalg.norm(wraw), 'wlin norm', np.linalg.norm(wlin))\nf=np.load('feats_drop46.npz'); M0=f['M0']\nprint('std of carry raw', ((M0-mu0)@wraw).std(0).round(4))\ng1=W(P+'attention.output.LayerNorm.weight')[RES]; b1=W(P+'attention.output.LayerNorm.bias')[RES]; g2=W(P+'output.LayerNorm.weight')[RES]; b2=W(P+'output.LayerNorm.bias')[RES]\nprint('L0 LN RES', g1.round(3), b1.round(3), g2.round(3), b2.round(3))"
}
```

