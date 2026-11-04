## reasoning summary

I'm checking the scale of the standardized values, which look fine, and noting that dimension 308 has a large mean that z-scoring takes care of. The bias term comes out to a reasonable magnitude too. I'm also checking float32 precision on the pre-activation, where that large input value and the weights stay manageable.

I'm now writing the edits for layer 11 and the pooler, with the pooler rows set to subtract the mean across the residual dimensions. After that I'll compute the expected logits in both float64 and a float32 simulation.



The weights are checked out, and I'm writing the head into layer 11, then saving the expected logits to compare against a harness read.



## tool_use: run_python

```json
{
 "code": "exec(open('krr.py').read())\nfrom scipy.special import erf\ndef gelu(a): return 0.5*a*(1+erf(a/np.sqrt(2)))\ny=np.load('base_feats.npz')['y']; X1=np.load('X1_full.npy'); F=X1[:,KEEP]; n=len(y); dd=F.shape[1]\nmu=F.mean(0); sd=F.std(0)+1e-6; Z=(F-mu)/sd\nlam=0.01\nK=gelu(Z@Z.T/dd); T=np.eye(8)[y]; T=T-T.mean(0)\nal=np.linalg.solve(K+lam*np.eye(n),T)\nW1=np.zeros((3072,768)); b1=np.zeros(3072)\nW1[:n][:,KEEP]=Z/(sd*dd)        # row i: z(c_i)/(sd*d)\nb1[:n]=-(Z/(sd*dd))@mu\nW2=np.zeros((768,3072)); W2[RES[:,None], np.arange(n)[None,:]]=al.T\n# check preact identity\npre=X1@W1[:n].T+b1[:n]; print('preact err', np.abs(pre-Z@Z.T/dd).max())\nsave_edit('bert.encoder.layer.11.intermediate.dense.weight', W1.astype(np.float32))\nsave_edit('bert.encoder.layer.11.intermediate.dense.bias', b1.astype(np.float32))\nsave_edit('bert.encoder.layer.11.output.dense.weight', W2.astype(np.float32))\nsave_edit('bert.encoder.layer.11.output.dense.bias', np.zeros(768,np.float32))\nsave_edit('bert.encoder.layer.11.output.LayerNorm.weight', np.ones(768,np.float32))\nsave_edit('bert.encoder.layer.11.output.LayerNorm.bias', np.zeros(768,np.float32))\npw=W('bert.pooler.dense.weight').copy(); pb=W('bert.pooler.dense.bias').copy()\na=1.0\npw[:8]=0; \nfor k in range(8): pw[k,RES]=a*(np.eye(8)[k]-1/8)\npb[:8]=0\nsave_edit('bert.pooler.dense.weight', pw.astype(np.float32)); save_edit('bert.pooler.dense.bias', pb.astype(np.float32))\n# expected logits in float32 simulation\ndef ln(X,g,b,eps=1e-12):\n    m=X.mean(1,keepdims=True); v=((X-m)**2).mean(1,keepdims=True); return (X-m)/np.sqrt(v+eps)*g+b\nf32=lambda t: t.astype(np.float32)\nH=gelu(f32(X1)@f32(W1).T+f32(b1)); V=f32(X1)+H@f32(W2).T\nout=ln(V,1.0,0.0); logits=np.tanh(out@f32(pw).T+f32(pb))[:,:8]\ns=K@al\nprint('argmax agree with KRR', (logits.argmax(1)==s.argmax(1)).mean(), 'train acc', (logits.argmax(1)==y).mean())\nsrt=np.sort(logits,1); print('min top1 margin', (srt[:,-1]-srt[:,-2]).min(), 'logit range', logits.min(), logits.max())\nnp.save('expected_logits.npy', logits)\n"
}
```

