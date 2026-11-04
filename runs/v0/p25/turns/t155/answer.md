## reasoning summary

I'm settling on w=0.4 and lam=0.005, since everything sits flat around 0.994–0.995. For the v5 build, the centers will be 1496 practice samples plus 1576 of mine picked at random with a fixed seed, and training will use all 3442. I considered favoring the newest my5 or the hardest cases, but random looks fine, so I'll build and save it, then compute expected logits for the practice set and all of mine.



Results are flat (0.994–0.995), so I'm going with weight 0.4 and lambda 0.005. Building v5 with 3072 neurons on all 3442 points next, then saving it and computing expected logits.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0,'.')\nfrom kh import *\nD=load(); d5=np.load('my5_feats_clean.npz')\nZp,yp=D['Zp'],D['yp']\nZmine=np.concatenate([D['Zm'],D['Z3'],D['Z4'],d5['Z']]); ymine=np.concatenate([D['ym'],D['y3'],D['y4'],d5['y']])\nmu,sd,nbar=D['mu'],D['sd'],np.load('kernel_v3.npz')['nbar']\nNC=3072-1496\nsel=np.sort(np.random.default_rng(7).permutation(len(Zmine))[:NC])\nC=np.concatenate([Zp,Zmine[sel]])\nZt=np.concatenate([Zp,Zmine]); yt=np.concatenate([yp,ymine]); wt=np.concatenate([np.ones(1496),np.full(len(Zmine),0.4)])\nlam=0.005\nPhi=gelu(Zt@C.T/768); A,b0=fit(Phi,yt,wt,lam)\nF=Phi@A+b0\nprint(C.shape, np.abs(A).max(), 'train acc all', (F.argmax(1)==yt).mean(), 'prac', (F[:1496].argmax(1)==yp).mean(), 'mine', (F[1496:].argmax(1)==ymine).mean())\nrng=np.random.default_rng(0); Fn=gelu((Zt[:500]+rng.normal(0,1e-3,(500,768)))@C.T/768)@A+b0; print('noise sens', np.abs(Fn-F[:500]).max())\nnp.savez('kernel_v5.npz',mu=mu,sd=sd,C=C,sel=sel,Zt=Zt,yt=yt,wt=wt,lam=lam,A=A,b0=b0,F=F,nbar=nbar)\nU=np.load('U_v2.npy'); S=100\np='bert.encoder.layer.11.'\nW1=(C/sd/768); b1=-(W1@mu)\nW2=S*U@A.T; b2=U@(S*b0-U.T@nbar)\nsave_edit(p+'intermediate.dense.weight',W1.astype(np.float32))\nsave_edit(p+'intermediate.dense.bias',b1.astype(np.float32))\nsave_edit(p+'output.dense.weight',W2.astype(np.float32))\nsave_edit(p+'output.dense.bias',b2.astype(np.float32))\nn=Zt*sd+mu\nx=n+(S*F-(U.T@nbar))@U.T\nxm=x-x.mean(1,keepdims=True); ln=xm/np.sqrt((xm**2).mean(1,keepdims=True)+1e-12)\nlogits=10*np.tanh(0.05*ln@U)\nprint('expected acc', (logits.argmax(1)==yt).mean())\nnp.save('v5_expected_logits.npy',logits)\nopen('notes.md','a').write('''- my5: my_sents5_all.json (461), base L11 feats pass_15 -> my5_feats.npz; cleaned (5 ambiguous dropped) -> my5_feats_clean.npz (456). v4 acc on my5 = .963.\n- CV: v4 config .9924; +my5 as non-centers .9942; as centers .9951 (can't fit: 3442>3072). Random vs pivoted-Cholesky center selection same (.994).\n- v5 SAVED (run142): 3072 neurons = 1496 prac + 1576 random(seed7) of mine(1946); trained on all 3442 (w .4 mine, lam .005). kernel_v5.npz, v5_expected_logits.npy (rows = Zt order: prac then m,m3,m4,m5clean). Next: verify with predict pass.\n''')\n"
}
```

