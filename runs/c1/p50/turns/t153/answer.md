## reasoning summary

I'm merging the features into the feats file for all 1496 samples, then running the honest estimates on the full set: the K9 and linear M0 joint model, with 10-fold over five seeds and 20-fold. I'll compare those against the full-set baseline of 98.46.



All 1,496 sentences are read, and I'm merging the features and running the full-set estimate next, with the base reference at 98.46.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom lib import labels, krr_cv, KEEP, RES\ny=labels()\nF={}\nfor nm in ['mean_hidden_L0','cls_hidden_L0','mean_hidden_L9','cls_hidden_L9']:\n    F[nm]=np.concatenate([np.load(f'pass_34/{nm}.npy'),np.load(f'pass_35/{nm}.npy')]).astype(np.float32)\nnp.savez('feats_seq1024.npz', **F)\nM9=F['mean_hidden_L9'].astype(np.float64); M0=F['mean_hidden_L0'].astype(np.float64)\ndef cosK(X,g):\n    Xc=X-X.mean(0); Xn=Xc/np.linalg.norm(Xc,axis=1,keepdims=True); S=Xn@Xn.T; return np.exp(g*(S-1))\ndef lin(X):\n    Xc=X-X.mean(0); return Xc/np.sqrt((Xc**2).sum(1).mean())\nK9=cosK(M9[:,KEEP],1.0); L0=lin(M0)\nprint('K9', krr_cv(K9,y,[0.01],k=10,seeds=range(5)))\nfor c in [0.25,0.5,1.0]:\n    print('K9+linM0',c, krr_cv(K9+c*L0@L0.T,y,[0.003,0.01,0.03],k=10,seeds=range(5)))\nprint('20f', krr_cv(K9+0.5*L0@L0.T,y,[0.01],k=20,seeds=range(3)))\nwith open('notes.md','a') as f:\n    f.write('''- Sequential re-prune DONE (L5 .070, L6 .080, L7 .077, L8 .080 heldout err). structure = layers 0..8, m=1024 seq keeps (keep_seq1024.json).\n- pass_34 (p0000-p1023) + pass_35 (p1024-p1495): edited cls/mean hidden L0,L9 -> feats_seq1024.npz.\n- 1024-subset honest: K9 97.1, K9+linM0 97.97 (old m1024 97.91, m1536 98.09, base 98.16). Carry RES corr .98 at L9 (carried RES CV = leaky 99.8, ignore).\n- CLS in kernel / g!=1 doesn't help.\n''')\n"
}
```

