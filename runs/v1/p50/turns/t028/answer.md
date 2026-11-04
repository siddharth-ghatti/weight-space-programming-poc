## reasoning summary

I'm puzzled that the R coordinates are nonzero in m10 and c12, since the carry heads have zero value and should send nothing into R. Setting the head's value rows and bias to zero should make its output zero, and the other heads' rows for R are zeroed too, so I want to find where the nonzero values come from.

I'm checking that the R coordinates come out equal across all eight in c12, since the shift from the layer-norm mean is the same for each. That common offset cancels in argmax, so it shouldn't change predictions. Next I'm setting up the final ridge fit on the four feature sets with lambda around 200 to 300, then converting the weights back to raw units.

I'm checking whether the per-sentence variation in the carry scaling, a_i ranging about ±9%, can be removed, since the in-sample fit assumed a single exact scale. The question is whether the product of layer-norm factors along the path can be made uniform.

I'm working through a way to cancel the per-sentence scaling by carrying a constant reference coordinate alongside the R coordinates, so the scale factor a_i shows up in both and could be divided out. The catch is that the pooler's tanh gives no division, so that route doesn't remove the variation.

I'm checking whether the per-sentence variation in the carry scale can be avoided. Since the final scale comes from averaging layer-norm factors across tokens and then applying the final norm, it varies by about ±9% per sentence. An earlier simulation that normalized by the mean showed this variation didn't hurt accuracy. I'm considering fitting the final classifier directly on the actual carried values after installing the heads, so the real scale is measured rather than assumed.

I'm settling on the joint fit as it stands and decoding the R coordinates by 1/(gw·ā), with a read afterward to check it. The per-sentence scale variation looks harmless because c12's non-R part is already unit-normalized, and I'm thinking about picking gw around 0.2 so the carried values end up at roughly the same magnitude as the logits.

I'm settling on gw around 0.1 so the carried values stay small next to the other features, and I'm expecting only a small shift in the final layer-norm scale. The common offset across the R coordinates is still harmless for argmax, and I'm about to check the pooler and classifier side.

I'm setting the pooler weights so each of the eight outputs is the scaled joint-fit logit, with the R part as an identity map divided by gw·ā, and the classifier as a plain identity on those first eight dimensions. Using s_p = 0.5 keeps tanh in its near-linear range, and since tanh is monotone, argmax is unaffected. The leftover constant only shifts all eight logits equally, so it doesn't matter. Then I'll verify with predictions on all 1496 sentences, in three batches of at most 512.

Setting up the install and checking the logits, I'm choosing a ridge penalty of 300 over 200 for a bit more regularization, since the CV curve is flat anyway. Then I'll read logits and the layer-12 CLS hidden state for all 1496 sentences to measure the scale and verify.



Joint ridge on the real features reaches 97.9% CV at λ=200–300, matching the simulation, and R is just the expected LayerNorm shift, which I'll verify next. I'm now running the final fit at λ=300 on all 1496 sentences and installing the weights into the carry heads and pooler rows.



## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'build2.py').read())\nlab=np.load(WORK/'labels.npy'); N=len(lab); Y=np.eye(8)[lab]\nF=dict(np.load(WORK/'prep_feats.npz'))\nr=F['c12'][:,R]; print('R coords equal across 8? max spread', (r.max(1)-r.min(1)).max())\nparts=['m2','m6','m10','c12']; lam=300.0\nX=np.concatenate([F[p][:,KEEP] for p in parts],1)\nmu=X.mean(0); sd=X.std(0)+1e-6; Z=(X-mu)/sd\nWz=np.linalg.solve(Z.T@Z+lam*np.eye(Z.shape[1]), Z.T@(Y-Y.mean(0)))\nWu=Wz/sd[:,None]; const=-(mu/sd)@Wz+Y.mean(0)\npred_in=(X@Wu+const); print('in-sample acc', (pred_in.argmax(1)==lab).mean())\nblocks={}\nfor i,p in enumerate(parts):\n    B=np.zeros((768,8),np.float32); B[KEEP]=Wu[i*760:(i+1)*760]; blocks[p]=B\nnp.savez(WORK/'joint_v2.npz', **blocks, const=const, mu=mu, sd=sd, lam=lam)\ngw=0.1; abar=4.6   # provisional decode scale; verify with read\nDs={2:(blocks['m2'].T,np.zeros(8)),6:(blocks['m6'].T,np.zeros(8)),10:(blocks['m10'].T,np.zeros(8))}\nsp=0.5\nWfin=blocks['c12'].T.copy()   # [8,768]\nfor k in range(8): Wfin[k,R[k]]=1.0/(gw*abar)\npw=np.zeros((768,768),np.float32); pb=np.zeros(768,np.float32)\npw[:8]=sp*Wfin; pb[:8]=sp*const\ncw=np.zeros((8,768),np.float32); cw[:,:8]=np.eye(8); cb=np.zeros(8,np.float32)\nE=build(Ds=Ds,gw=gw,head_cls=(pw,pb,cw,cb))\nprint(len(edits()),'edits saved')\n# expected carried sum magnitude\ncarried=sum(F[p]@blocks[p] for p in ['m2','m6','m10']); print('carried logits range', carried.min().round(2), carried.max().round(2), 'acc', (carried.argmax(1)==lab).mean())\nnp.save(WORK/'expected_logits_v2.npy', pred_in)\n"
}
```

