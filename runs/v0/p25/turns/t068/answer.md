## reasoning summary

Several of these disagreements look like plain model mistakes, such as coverage being marked wrong for sentences saying custodial care and over-the-counter medications are not covered, and the deductible being mislabeled for a premium amount. The model seems weaker on my sentence style, which is shorter and more generic, with 92% accuracy on my examples, so I'm wondering whether the test set matches the practice distribution.

The model's mistakes look like plain errors on negation sentences and mislabeled premium amounts, but my own sentences are shorter and more generic, and the model does worse on them at 92%. Since the test set should match the practice distribution, practice CV is the criterion that counts. Adding all my sentences left practice CV unchanged while improving coverage elsewhere, so I'm weighing whether to include them all, include only the ones I'm confident about, or downweight them.

I'm weighing whether to drop the ambiguous sentences from my own set or keep all of them, since removing them by hand is a judgment call that the practice data itself doesn't settle. Filtering to the ones the full practice model agrees with gives a cleaner comparison, at 498 extra sentences and well under the 3072-center budget. I also want to test a lower weight on my sentences and see whether more data would keep helping.

I'm thinking the practice sentences are longer and more specific, with company names and letters, so writing more in that style might help more than my own shorter ones, though the gain looks small, around 0.2 to 0.3 percent. A different direction, improving the features, could have a bigger effect, but I haven't worked that out yet.

I'm planning to build a third version that adds the sentences I agree with as extra centers, bringing the total to about 1994, and then check the regularization. After that I'd run a quick prediction pass on the practice set, which should come out near 1.0, plus a read on my own sentences.

I'm now weighting the rows of the ridge solve, testing my sentences at weights of 0.3 and 0.5 with the full set, and at 0.5, 1, and 2 with the agreeing subset, with the centering done using weighted means.



The model's errors look like real misclassifications, not labeling problems, so it trails on my shorter phrasing (92%) more than on practice text (98%). Since the test set follows the practice distribution, practice CV stays the deciding metric, and I'm next testing sample weighting and lambda under honest per-fold filtering.



## tool_use: run_python

```json
{
 "code": "import json\nfrom scipy.special import erf\nf=np.load('base_feats.npz'); y=f['y']\ndef gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))\ndef LN(x): \n    m=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-m)/np.sqrt(v+1e-12)\nA=json.load(open('my_sents_all.json')); ym=np.array(A['labels'])\nXp=LN(f['cls11']+2*f['mean11'])\nXm=LN(np.load('pass_10/cls_hidden_L11.npy').astype(np.float64)+2*np.load('pass_10/mean_hidden_L11.npy'))\nmu=Xp.mean(0); sd=Xp.std(0)+1e-6\nZp=(Xp-mu)/sd; Zm=(Xm-mu)/sd; d=768\ndef fitw(Ztr,ytr,w,lam=0.01):\n    P=gelu(Ztr@Ztr.T/d); T=np.eye(8)[ytr]; ws=w/w.sum()\n    Pm=ws@P; Tm=ws@T; Pc=P-Pm\n    A_=np.linalg.solve(Pc.T@(Pc*w[:,None])+lam*np.eye(len(Pc)), Pc.T@((T-Tm)*w[:,None]))\n    return lambda Z: (gelu(Z@Ztr.T/d)-Pm)@A_+Tm\ndef cvrun(seed, mode, wm=1.0, lam=0.01):\n    rng=np.random.default_rng(seed); perm=rng.permutation(len(y)); fo=np.zeros(len(y),int); fo[perm]=np.arange(len(y))%5\n    pred=np.zeros(len(y),int)\n    for k in range(5):\n        tr=fo!=k; ntr=tr.sum()\n        if mode=='agree':\n            g0=fitw(Zp[tr],y[tr],np.ones(ntr),lam); mm=np.argmax(g0(Zm),1)==ym\n        else: mm=np.ones(len(ym),bool)\n        Ztr=np.vstack([Zp[tr],Zm[mm]]); ytr=np.concatenate([y[tr],ym[mm]])\n        w=np.concatenate([np.ones(ntr), wm*np.ones(mm.sum())])\n        g=fitw(Ztr,ytr,w,lam); pred[~tr]=np.argmax(g(Zp[~tr]),1)\n    return (pred==y).mean()\nfor mode,wm,lam in [('all',0.3,0.01),('all',0.5,0.01),('agree',0.5,0.01),('agree',1,0.01),('agree',2,0.01),('agree',1,0.005),('agree',1,0.02)]:\n    r=[cvrun(s,mode,wm,lam) for s in range(8)]\n    print(mode,wm,lam, round(np.mean(r),4))\n"
}
```

