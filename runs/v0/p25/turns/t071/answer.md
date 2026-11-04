## reasoning summary

I'm confirming the structure, and noting that the key k uses the same mean and standard deviation as my earlier Xp-based computation. Next I want to check whether k['Z'] matches my Zp, which I'll do by looking at the build script. Then I'm assembling v3 with centers from Zp and Zm, giving 2039 in total, and setting the weights to 1 for the practice items.

Now I'm working out the expected logits for the practice items and my sentences, using the closed-form head: the pooled output is a tanh of the projected, layer-normed sum of the activations and the edited correction.

Now I'm checking whether the float32 precision of the large W2 coefficients causes trouble, since lowering lam makes A bigger and the cancellation across 2039 neurons could hurt. I'll compare the float32 forward pass against float64 to see how much error there is.

Now I'm checking the float32 kernel predictor against float64 on the read activations, since the closed-form head is just my own edited classifier evaluated on those activations and not a fresh forward pass.



I've confirmed the v2 structure, so I'll build v3 the same way with 2039 kernel centers, sentence weight 0.3 and lambda 0.005, checking the numpy results and float32 sensitivity before saving.



## tool_use: run_python

```json
{
 "code": "import json\nfrom scipy.special import erf\nf=np.load('base_feats.npz'); y=f['y']\ndef gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))\ndef LN(x): \n    m=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-m)/np.sqrt(v+1e-12)\nk=np.load('kernel_v2.npz'); U=np.load('U_v2.npy')\nA=json.load(open('my_sents_all.json')); ym=np.array(A['labels'])\nXp=LN(f['cls11'].astype(np.float64)+2*f['mean11'])\nXm=LN(np.load('pass_10/cls_hidden_L11.npy').astype(np.float64)+2*np.load('pass_10/mean_hidden_L11.npy'))\nmu=k['mu']; sd=k['sd']\nprint('mu/sd match', np.abs(mu-Xp.mean(0)).max(), np.abs(sd-(Xp.std(0)+1e-6)).max(), np.abs(k['n']-Xp).max())\nZp=(Xp-mu)/sd; Zm=(Xm-mu)/sd; d=768\nC=np.vstack([Zp,Zm]); yc=np.concatenate([y,ym]); w=np.concatenate([np.ones(len(y)),0.3*np.ones(len(ym))])\nlam=0.005\nP=gelu(C@C.T/d); T=np.eye(8)[yc]; ws=w/w.sum(); Pm=ws@P; Tm=ws@T; Pc=P-Pm\nAw=np.linalg.solve(Pc.T@(Pc*w[:,None])+lam*np.eye(len(Pc)), Pc.T@((T-Tm)*w[:,None]))\nb0=Tm-Pm@Aw\nF=P@Aw+b0\nprint('train acc practice', (np.argmax(F[:1496],1)==y).mean(), 'mine', (np.argmax(F[1496:],1)==ym).mean())\nprint('|A| max', np.abs(Aw).max(), 'v2 |Wf| max', np.abs(k['Wf']).max())\nS=100.\nnbar=Xp.mean(0)\ndef expected_logits(Xn, Fv):\n    x=Xn + (S*Fv - (U.T@nbar))@U.T\n    ln=(x-x.mean(1,keepdims=True))/np.sqrt(x.var(1,keepdims=True)+1e-12)\n    return 10*np.tanh(0.05*ln@U)\nEp=expected_logits(Xp,F[:1496]); Em=expected_logits(Xm,F[1496:])\nprint('expected acc', (Ep.argmax(1)==y).mean(), (Em.argmax(1)==ym).mean())\n# float32 sensitivity of FFN\nW1=(C/sd/d).astype(np.float32); b1=(-(C/sd/d)@mu).astype(np.float32)\nW2=(S*U@Aw.T).astype(np.float32); b2=(U@(S*b0-U.T@nbar)).astype(np.float32)\nh=gelu((Xp.astype(np.float32)@W1.T+b1).astype(np.float64)).astype(np.float32)\no32=(h@W2.T+b2).astype(np.float64)\no64=(S*F[:1496]-U.T@nbar)@U.T\nprint('float32 err in FFN out (proj on U)/S', np.abs((o32-o64)@U).max()/S)\nnp.savez('kernel_v3.npz', mu=mu, sd=sd, C=C, yc=yc, w=w, lam=lam, A=Aw, b0=b0, F=F, nbar=nbar)\nnp.save('v3_expected_logits_practice.npy', Ep); np.save('v3_expected_logits_mine.npy', Em)\n"
}
```

