## reasoning summary

I'm working out the scale of the token class scores, which runs about 0.8 to 1.1 within a sentence. I'm estimating that a beta around 2 to 3 would make attention peaky but still soft. Next I'm thinking about fitting a ridge regression from the token vectors to an exponentially decaying position target, while leaving out the CLS and SEP tokens.

I'm weighing a ridge readout that maps each token vector to a linear position score, then using a decaying exponential over that score to pool tokens by distance from the start. The open question is CLS at position 0, which would get the largest weight under that scheme, so I'll need to exclude it from the pools.

I'm adding a strong negative score for the CLS token along its own embedding direction, so it doesn't dominate the position pool. I'd like to check how that affects attention on the tok64 sentences. I'm also considering a simpler design with 12 heads: eight class-based pools at β=2.5, and one position-decay head.

Still setting up the value side: I'm weighing the eight class directions plus a PCA basis from the same token data, and I haven't settled what goes in the remaining slots.

I'm settling on the value side now: each head's output should land on a similar scale so the layer norm doesn't let one block dominate, so I'll normalize each head's value projection to roughly unit RMS.

I'm checking whether the LN mean-subtraction across all 768 dimensions mixes the per-head blocks, since that would make the block-identity readout only approximately recoverable. I'm considering whether setting the output bias just right would cancel that centering, so the head outputs come through cleanly for this evaluation.

I'm checking whether the layer norm's mean subtraction across all 768 dimensions would undo the per-head block identity. Shifting one dimension by a large offset would only shrink everything in float32, so I'll skip that. Instead I'll read the features as they are, with the blocks re-normalized offline, and set the final layer norm to unit gain and zero bias so the output is essentially the normalized vector.

I'm checking how the attention scores are computed in the HuggingFace BERT layer, specifically the scaling by the square root of the head dimension and the padding mask added before the softmax, so the pooling weights come out as I expect.

I'm checking the attention arithmetic in the BERT layer: scores are scaled by the square root of the head dimension, padding is masked before the softmax, and each sublayer ends with a residual plus layer norm. Next I'm planning the E1 builder, starting with a projection that stacks eight class directions and top PCA components from the token data into 64 value dimensions per head.

Still building the head weights: I'm setting the query to zero and the key's first row to the scoring direction, so each head's score is just that direction dotted with the token vector, and I'm working out how to scale the class-head directions.

I'm setting the early-position score to the negative of the decayed position direction, with a penalty on the CLS direction, and I'll check that CLS scores lower than the other tokens. The uniform head gets a zero direction, and the late-position head uses the positive decay direction. Each head's value projection is fixed, and the output projection is an identity block with the CLS vector subtracted to cancel the residual. I'm noting that the per-vector mean from layer norm won't be affected by this subtraction in any way I can plan for now, so I'll handle the centering offline.

I'm making the final LayerNorm and output-side weights identity-like, with the FFN output zeroed out, so the readout is essentially the normalized attention result. The layer-centering effect on the per-head blocks is only approximate, which I'll accept for this experiment.



I'm building experiment E1 on original layer 11 with all 12 heads rewritten as pools, then fitting the position direction and checking how to suppress CLS next.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\nmask=np.load('pass_4/attention_mask.npy').astype(bool); X0=np.load('pass_4/token_hidden_L0.npy')\nT=X0.shape[1]; lens=mask.sum(1)\npos=np.tile(np.arange(T),(64,1))\nissep=np.zeros_like(mask); issep[np.arange(64),lens-1]=True\nbody=mask.copy(); body[:,0]=False; body&=~issep\nXb=X0[body]; pb=pos[body]\nxcls=X0[0,0]; print('cls const?', np.abs(X0[:,0]-xcls).max())\n# position direction: ridge x->pos on body tokens\nmu=Xb.mean(0); A=Xb-mu; lam=10.0\nr=np.linalg.solve(A.T@A+lam*np.eye(768),A.T@(pb-pb.mean()))\npred=A@r; print('pos R2', 1-((pred-(pb-pb.mean()))**2).mean()/pb.var())\nprint('pred vs pos (mean per pos):', [round(float(pred[pb==p].mean()+pb.mean()),1) for p in [1,2,3,5,8,12,20,30,40]])\n# r . x for cls and sep\nprint('r.x cls', float((xcls-mu)@r+pb.mean()), 'sep', float(((X0[issep]-mu)@r).mean()+pb.mean()))\n# cls suppress direction\nc=xcls-Xb.mean(0); c/=np.linalg.norm(c)\nprint('c.x cls',float(c@xcls),'c.x body pct',np.percentile(Xb@c,[1,50,99]),'sep',np.percentile(X0[issep]@c,[1,50,99]))\n# token PCA\nXall=X0[mask]; U,S,Vt=np.linalg.svd(Xall-Xall.mean(0),full_matrices=False); print('pca var top', (S[:64]**2/ (S**2).sum()).sum())\nnp.savez('pool_design.npz', **dict(np.load('pool_design.npz')), r=r, rmu=mu, pmean=pb.mean(), c=c, xcls=xcls, pca=Vt[:128], tokmean=Xall.mean(0))"
}
```

