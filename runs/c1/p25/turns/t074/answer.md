## reasoning summary

The network matches exactly, and the k=10 model reproduces the target: 100% in-sample accuracy and logits matching tanh(cen F) to within 1e-5. The cross-validation estimate sits around 98.1. I'm double-checking the margin figures next: a minimum top-1 margin of 0.77 looks odd beside a maximum absolute logit of 0.72, but since margin is top-1 minus top-2, values like 0.7 and -0.07 make that consistent. So this k=10 model, at 95.3M parameters, is a valid submission candidate.

Saving a backup of the k=10 edits and structure comes first. Then I'll explore k=9 with richer pooling, aiming for a CV of at least 98.0 under the M8-based setup, with a budget of 14 passes.

I'm weighing whether the pooling could be built into the attention heads themselves: a head per class that attends to tokens scoring high on that class, with the value carrying the token states. The catch is that the CLS output is a single 768-dimensional stream, so any head repurposed for this gives up its share of the uniform mean that's currently being used.

I'm working out whether the 512 PCA dimensions of the M8 mean can carry the pooled signal, since eight heads at 64 dimensions each would cover that, leaving four heads free for other pooling. I'm also considering keeping the CLS residual as an extra input, and I want to check the CV for a PCA-512 kernel before committing to that split.

I'm checking whether the 512 PCA components of the M8 mean keep the signal, since that would free four heads for other pooling. I'm also weighing class-specific soft-max pooling over token-level scores, which would need a test on the token hidden states before committing to it.

I'm weighing whether pooling designs could add information beyond the uniform mean of hidden layer 8. Class-specific max-style pooling over token scores is speculative, since per-token scores may be noisy and special tokens like CLS and SEP could dominate. Excluding those tokens from the mean would likely make little difference, so the bigger question is whether any extra pooling is worth the budget.

I'm considering whether per-dimension weighting of the M8 features, such as whitening with shrinkage, could beat the plain z-scored kernel. I'll test a few of those variants in numpy before deciding whether the extra pooling designs are worth their budget.

I'm going to test a few variants of the M8 features in numpy first, since that's cheap: PCA-whitening with shrinkage, LDA-projected features feeding the kernel, and whether any signal can come from outside the M8 mean. Carry heads only pass linear features and each one takes 8 reserved dimensions, so adding more inputs would change the network itself, which I want to avoid.



The k=10 model matches tanh(centered F) to 1e-5 and scores about 98.1% in cross-validation, so I'll back it up as the fallback. Next I'll test cheap numpy kernel variants to see whether k=9 can reach the same CV.



## tool_use: run_python

```json
{
 "code": "import shutil, os\nos.makedirs('backup_k10',exist_ok=True)\nfor f in os.listdir('edits'): shutil.copy(f'edits/{f}', f'backup_k10/{f}')\nshutil.copy('structure.json','backup_k10/structure.json')\nopen('notes.md','a').write(\"\"\"- pass11 VERIFIED k=10 model: logits==tanh(cenF) to 1e-5, in-sample 100% (1024), min margin .77. BACKUP in backup_k10/ (edits + structure.json). CV ~98.1. 95,312,648 params. SAFE FALLBACK.\n- Next: explore k=9 ([0..7]+pool) readout improvements.\n\"\"\")\nimport sys; sys.path.insert(0,'.')\nfrom cvlib import folds\nfrom scipy.special import erf\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\nd=np.load('base_reads.npz'); y=d['y']; R=np.load('reserved.npy'); keep=np.setdiff1d(np.arange(768),R)\nM={j:d[f'mean_hidden_L{j}'][:,keep].astype(np.float64) for j in (2,6,8)}\ndef z(X): return (X-X.mean(0))/(X.std(0)+1e-6)\ndef kcv(K, lams, seeds=range(4)):\n    Y=np.eye(8)[y]; out={}\n    for lam in lams:\n        a=[]\n        for s in seeds:\n            f=folds(y,5,s); S=np.zeros((len(y),8))\n            for i in range(5):\n                tr=f!=i; te=f==i\n                A=np.linalg.solve(K[np.ix_(tr,tr)]+lam*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0))\n                S[te]=K[np.ix_(te,tr)]@A\n            a.append((S.argmax(1)==y).mean())\n        out[lam]=round(100*np.mean(a),2)\n    return out\nL26=z(M[2])@z(M[2]).T/760+z(M[6])@z(M[6]).T/760\nZ=z(M[8]); lams=(1e-2,3e-2,1e-1)\nprint('base gelu', kcv(2*gelu(Z@Z.T/760)+L26,lams))\nXc=M[8]-M[8].mean(0); Uu,S,Vt=np.linalg.svd(Xc,full_matrices=False)\nfor k in (128,256,512):\n    P=Xc@Vt[:k].T\n    for shrink in (0.3,1.0):\n        # partial whitening: scale pc by s^-shrink\n        W=P/(S[:k]**shrink); W=W/np.sqrt((W**2).sum(1).mean())*1.0\n        T=W@W.T\n        print('pca',k,'whiten^',shrink, kcv(2*gelu(T)+L26,lams))"
}
```

