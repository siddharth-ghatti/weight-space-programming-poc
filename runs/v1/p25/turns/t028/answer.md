## reasoning summary

The separation looks clean: the token dot products sit near zero, and the CLS score is essentially 1, so the perturbation to the tokens' queries is only around 1.5% of the CLS query's magnitude. Next I'm applying the tap to every head in layer 6 at once, subtracting the outer product of the mean CLS query and the direction vector from the query weights.

I'm working out what the CLS query becomes after the tap: it reduces to roughly W_q δ, so it's a deviation-driven query at about 15% of normal magnitude. Attention should be fairly flat there rather than uniform. I'm weighing whether to project out the deviation subspace, but that would also disturb the token components, so I'll leave it and go with mostly-flat attention, applied to all heads. Next I plan to read the layer 7–11 hidden states to see how the tokens shift.

I'm weighing whether to tap layer 6 or move to layer 4 or 5, but the token data I have is only at layer 6, so I'll stick with it for now and plan to try tapping later layers in follow-up experiments.



Separation looks excellent (token d·h ≈ 0, [CLS] d·h = 1.000), so the tap leaves other tokens' queries almost unchanged; I'm applying it to all 12 heads of layer 6 and then measuring how much of the injected signal survives through layers 7–11.



## tool_use: run_python

```json
{
 "code": "F=np.load('feats.npz'); T=np.load('tok128.npz'); sid=T['sid']\nfirst=np.r_[True, sid[1:]!=sid[:-1]]\nH6=T['T6'][~first]; C6=F['C6']; Cbar=C6.mean(0); D=C6-Cbar\nS=H6.T@H6/len(H6) + D.T@D/len(D) + 0.01*np.eye(768)\nd=np.linalg.solve(S,Cbar); d/= d@Cbar\nnp.save('d_L6.npy',d)\np='bert.encoder.layer.6.attention.self.'\nWq=base(p+'query.weight'); bq=base(p+'query.bias')\nqbar=Wq@Cbar+bq\nWq2=(Wq-np.outer(qbar,d)).astype(np.float32)\nsave_edit(p+'query.weight',Wq2)\nprint('qbar norm per head',np.linalg.norm(qbar.reshape(12,64),axis=1).round(1))\nprint(sorted(edits()))\nopen('notes.md','a').write(\"\\n- Baseline readout (base C11+M11, wc=.7, lam=300, c=100) saved & verified exact (pass 7). CV est ~96.7%. Saved readout_base.npz.\\n- Exp A: tap layer6 all heads: Wq' = Wq - qbar d^T (d_L6.npy; tok d.h ~0+-.015, cls 1+-.009). Reading C7..C11', M7..M11' (passes 8-9).\\n\")"
}
```

