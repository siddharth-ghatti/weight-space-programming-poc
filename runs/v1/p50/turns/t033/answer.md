## reasoning summary

The scaling makes no real difference to accuracy, with everything landing around 97.8%, so the approach looks robust. The installed model still uses a uniform weight of 0.1 across all layers, which leaves the per-layer balance off: layer 10's contribution ends up roughly half-weighted compared with layer 2.

I'm settling the refit: I'll fit the weights on the scaled features and install them with per-layer gains of gw divided by each layer's mean, then decode with weight 1/gw. Since the installed value weights are gw times D_l, D_l has to be W_l divided by the layer's mean, which keeps everything consistent. The common sentence-dependent shift in the coordinates doesn't affect anything.

I'm weighing a few further ideas now: whether leaving out the special tokens or punctuation from the mean pooling helps, and whether a shrinkage LDA on the joint features beats plain ridge. I'd rather not emulate the network's feed-forward layer in numpy, so I'm setting that option aside.

I'm considering whether a token-level weighting of the pooling, such as down-weighting function words or emphasizing content tokens, could help beyond the mean. I can't test that yet because token-level reads are capped at 64, so it stays a hypothetical for now. I'm also feeling that 97.8% is probably near the ceiling for this approach, with the remaining errors looking like genuinely ambiguous labels.

I'm ruling out the token-level pooling idea, since token-level reads are capped and the cost is too high. I also see no way to add more heads in the final aggregator with only one read path, so I'm dropping that too. Now I'll finalize the v3 version: refit on the scaled features, install it, and verify.



Scaling leaves accuracy unchanged at about 97.8%, but the installed model under-weights layers 10 and 6, so I'm refitting on scaled features with per-layer value gains and checking that the logits match numerically.



## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'build2.py').read())\nlab=np.load(WORK/'labels.npy'); N=len(lab); Y=np.eye(8)[lab]\nF=dict(np.load(WORK/'prep_feats.npz')); A=np.load(WORK/'v2_layer_scales.npy'); abar=A.mean(0)\nparts=['m2','m6','m10']; lam=250.0\nXs=np.concatenate([F[p][:,KEEP]*(A[:,k]/abar[k])[:,None] for k,p in enumerate(parts)]+[F['c12'][:,KEEP]],1)\nmu=Xs.mean(0); sd=Xs.std(0)+1e-6; Z=(Xs-mu)/sd\nWz=np.linalg.solve(Z.T@Z+lam*np.eye(Z.shape[1]), Z.T@(Y-Y.mean(0)))\nWu=Wz/sd[:,None]; const=-(mu/sd)@Wz+Y.mean(0)\nexp=Xs@Wu+const; print('in-sample acc', (exp.argmax(1)==lab).mean())\nblocks={}\nfor i,p in enumerate(parts+['c12']):\n    B=np.zeros((768,8),np.float32); B[KEEP]=Wu[i*760:(i+1)*760]; blocks[p]=B\ngw=0.1; sp=0.5\nDs={l:((blocks[p]/abar[k]).T, np.zeros(8)) for k,(l,p) in enumerate(zip([2,6,10],parts))}\nWfin=blocks['c12'].T.copy()\nfor k in range(8): Wfin[k,R[k]]=1.0/gw\npw=np.zeros((768,768),np.float32); pb=np.zeros(768,np.float32); pw[:8]=sp*Wfin; pb[:8]=sp*const\ncw=np.zeros((8,768),np.float32); cw[:,:8]=np.eye(8); cb=np.zeros(8,np.float32)\nE=build(Ds=Ds,gw=gw,head_cls=(pw,pb,cw,cb))\nnp.savez(WORK/'joint_v3.npz', **blocks, const=const, abar=abar, lam=lam, gw=gw, sp=sp)\n# expected logits (pre-tanh -> post): logits = tanh(sp*(exp + common shift terms)); common shift unknown, compare argmax & centered\nnp.save(WORK/'expected_v3.npy', exp)\nprint(len(edits()), 'edits saved')\nwith open(WORK/'notes.md','a') as f:\n    f.write(\"\"\"- v2 (pass_6,7 prepared reads -> prep_feats.npz; joint ridge lam300 [m2,m6,m10,c12] nonR; OOF 97.8%). Installed & verified (pass_8,9): network preds == numpy expected 100%.\n  Found per-layer R decode scales differ: a_l = [4.72, 3.58, 2.37] for carry layers 2,6,10 (per-sentence std 9/7/4%), saved v2_layer_scales.npy.\n- v3: refit on scaled feats (a_li/abar_l)*m_l (OOF 97.8-97.9%), install value gain gw/abar_l (gw=0.1), decode R with 1/gw, pooler rows0-7 = 0.5*[W_c12 | R decode], classifier = I on first 8. joint_v3.npz.\n\"\"\")\n"
}
```

