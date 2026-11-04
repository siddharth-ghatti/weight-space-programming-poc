## reasoning summary

I'm picking the eigenvectors with the smallest variance from the covariance of n, restricted to the subspace orthogonal to 1, and taking the 8 lowest. I also want to check that n·U doesn't pick up large values along those directions, which I can handle by subtracting the mean of Uᵀn.

Now I'm setting the layer 11 weights: the attention output projection becomes a scaled identity with zero bias, and the intermediate layer gets the feature rows with matching biases, zeroed elsewhere. I also confirm the residual path, since the feed-forward block receives the post-attention LayerNorm output.

Next I'm setting the pooler: its weights are zero except for the first eight rows, which take a small multiple of the U directions, with zero bias. Then the classifier gets a scaled identity over those eight outputs, also zero bias, so the argmax reads off the pooled features directly.

Now I'm checking the spread of U^T n against the gap S, to see whether the residual noise stays small next to the signal.



The coefficients are well-conditioned, so float32 noise barely moves scores against a 0.92 top-2 margin. Next I'll build the v2 edits using the 8 lowest-variance directions of n, orthogonal to the all-ones vector, and confirm n's leakage into U is small.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nK=np.load('kernel_v2.npz'); mu,sd,Z,Wf,b0,F,n=[K[k] for k in ['mu','sd','Z','Wf','b0','F','n']]\ny=np.load('base_feats.npz')['y']\nd=768; one=np.ones(d)/np.sqrt(d)\nPp=np.eye(d)-np.outer(one,one)\nC=np.cov((n@Pp).T)\nC=Pp@C@Pp\nev,evec=np.linalg.eigh(C)\n# drop the null direction (ones): pick smallest eigen dirs with |dot(one)| small\norder=[i for i in np.argsort(ev) if abs(evec[:,i]@one)<1e-3][:8]\nU=evec[:,order]; print('eigs',ev[order], 'U.1',np.abs(U.T@np.ones(d)).max(), 'orth',np.abs(U.T@U-np.eye(8)).max())\nproj=n@U; print('std of U^T n per dir',proj.std(0).round(4))\nS=100.0; beta=1.0\nnbar=n.mean(0)\nL='bert.encoder.layer.11.'\nW1=np.zeros((3072,d)); b1=np.zeros(3072)\nN=len(Z)\nW1[:N]=(beta/d)*(Z/sd)\nb1[:N]=-W1[:N]@mu\nW2=np.zeros((d,3072)); W2[:,:N]=S*U@Wf.T\nb2=U@(S*b0-U.T@nbar)\nsave_edit(L+'attention.output.dense.weight',(2*np.eye(d)).astype(np.float32))\nsave_edit(L+'intermediate.dense.weight',W1.astype(np.float32))\nsave_edit(L+'intermediate.dense.bias',b1.astype(np.float32))\nsave_edit(L+'output.dense.weight',W2.astype(np.float32))\nsave_edit(L+'output.dense.bias',b2.astype(np.float32))\neps=0.05\nWp=np.zeros((d,d)); Wp[:8]=eps*U.T\nsave_edit('bert.pooler.dense.weight',Wp.astype(np.float32)); save_edit('bert.pooler.dense.bias',np.zeros(d,np.float32))\nWc=np.zeros((8,d)); Wc[:,:8]=10*np.eye(8)\nsave_edit('classifier.weight',Wc.astype(np.float32)); save_edit('classifier.bias',np.zeros(8,np.float32))\nnp.save('U_v2.npy',U)\n# expected logits on practice (for verification)\nout=n+(S*F-(n@U)+ (n@U) - nbar@U)@U.T   # n + U s where s = S F - U^T nbar\nh=out/np.sqrt((out**2).mean(1,keepdims=True))\nexp_logits=10*np.tanh(eps*(h@U))\nprint('expected train acc', (np.argmax(exp_logits,1)==y).mean(), 'agree with F', (np.argmax(exp_logits,1)==np.argmax(F,1)).mean())\nnp.save('v2_expected_logits.npy',exp_logits)\nprint(sorted(edits()))"
}
```

