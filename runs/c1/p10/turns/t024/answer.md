## reasoning summary

Sample size seems to matter a lot here: even the base model drops from 98.2% to 97.2% on the 1024 subset. That points to a steep learning curve. Training the final model on all 1496 samples should push test accuracy above the 5-fold CV figure, which only trained on 1197, and that would account for the gap between the original CV of 97.8% and the test score of 98.3%.

The pruned model at m=1024 gets worse with depth, dropping from 95.8% at L1 to 94.4% at L12, and the relative change already exceeds 1 by the deeper layers, so the representation has been wrecked. Even at L1 the change is 0.27 from layer 0 alone, which makes me suspect the compensation step is what's hurting. Gaussian-proxy FFN pruning doesn't look viable at this width, so I need either a different approach or to drop the FFN pruning idea altogether.

I'm weighing whether the LN normalization makes the activation-reading trick messy, and whether reading attention outputs would count as reimplementing a sublayer, which I'd rather avoid. So I'm considering a different route: dropping whole layers instead, which means the next layer receives an earlier hidden state, and I need to figure out which layers could be removed without retraining.

I'm checking whether the compensation step is what's causing the damage: the relative change at layer 1 is already 27%, which seems too large if the kept units were chosen well. If the compensation were accurate, the drift should be smaller than that.

I'm weighing a different route now: since estimating the FFN input without activation statistics is hard, I'll set aside the pruning proxy and consider dropping whole layers instead. Some configurations keep layers 0 through 8 plus pooling, which is a safe baseline, and I'd test variants that remove middle layers from that range.

I'm weighing whether to keep the FFN pruning idea at all, since estimating its inputs without activation statistics is the hard part. Layer dropping stays on the table, and keeping layers 0 through 8 plus pooling is the baseline I'd compare against.

I'm wondering whether I can estimate the FFN input statistics from the token-level hidden states, even though attention contributes to that input. Using the layer's W1 and GELU on a crude proxy would be closer to computing importance statistics than running the model for predictions, but it's still a gray area. I'll weigh that against the layer-dropping route, which avoids the problem entirely.

I'm weighing a broader option: a two-layer model with a kernel on M1, boosted by the pooling-layer FFN for full kernel ridge regression. Its 5-fold CV is 97.4% on 1496 samples, but more training data has been pushing accuracy up, so a final model on all 1496 might land around 97.7–97.9%, which is borderline. I'd like to measure the learning curve more precisely with 10-fold CV, which trains on 1346.

I'm still weighing the economics of each option against the others. Pruning the FFN at m=2048 is the one idea not yet tested, so I'd like to see whether a gentler cut keeps the representation intact.

Maybe I could prune the FFN layers in a data-driven way by probing token hidden states at every layer, though computing the FFN input would mean reimplementing the attention sublayer, which I'd rather avoid.

I'm wary of computing FFN unit activations myself, since that would be running forward computation outside the forward_pass tool. Measuring unit importance by group ablation would need a full pass per configuration, which I can't afford. I could try a different ranking without compensation, or use weight-norm products instead, but each check costs a pass and might not help. I'll look at whether the FFN is really needed at this width before choosing.

I'm considering whether layer-dropping with a larger m is worth it, since FFN pruning hit a 27% drift at layer 1. Some reduced-layer configurations like k=7 gave weaker accuracy, so I'm thinking the learning-curve gain from more training data could make k=7 or k=9 acceptable. I'd like to quantify that gain with 10-fold CV, and the kernel ridge regression leave-one-out predictions can be computed in closed form for that.

I'm going to compute the leave-one-out error in closed form from the kernel hat matrix, using the residual divided by one minus the diagonal entry, so I get an accuracy estimate on all 1496 samples without running separate folds.



Proxy-based FFN pruning degrades the deep features badly, and the 1024-sentence subset understates accuracy compared with the full 1496. I'm computing exact leave-one-out accuracy for the candidate feature sets, since it's cheap for kernel ridge and better matches a head trained on all the data.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nexec(open(WORK/'lib.py').read())\nF = np.load(WORK/'base_feats.npz'); y=F['y']\nM = {L: F[f'mean_hidden_L{L}'][:,KEEP] for L in range(13)}\ndef zsall(X): mu=X.mean(0); sd=X.std(0)+1e-6; return (X-mu)/sd/np.sqrt(X.shape[1])\ndef loo(blocks, y, gammas=(0.2,0.4,0.8), lams=(0.001,0.003,0.01,0.03), lin=None, c=0.3):\n    Z=np.concatenate([zsall(X) for X in blocks],1)/np.sqrt(len(blocks)); sq=(Z**2).sum(1)\n    D2=np.maximum(sq[:,None]+sq[None]-2*Z@Z.T,0); T=np.eye(8)[y]-1/8; out={}\n    KL = 0 if lin is None else (lambda ZL: ZL@ZL.T)(np.concatenate([zsall(X) for X in lin],1)/np.sqrt(len(lin)))\n    for g in gammas:\n        K=np.exp(-g*D2)+c*KL; w,V=np.linalg.eigh(K)\n        for l in lams:\n            Hd=((V**2)*(w/(w+l))[None]).sum(1); Yh=V@((w/(w+l))[:,None]*(V.T@T))\n            Yloo=(Yh-Hd[:,None]*T)/(1-Hd[:,None])\n            out[(g,l)]=round(((Yloo).argmax(1)==y).mean(),4)\n    return out\nfor name,bl,lin in [('M0',[M[0]],None),('M1',[M[1]],None),('M0+M1',[M[0],M[1]],None),('M6',[M[6]],None),('M8',[M[8]],None),('M9',[M[9]],None),('M0+M8',[M[0],M[8]],None),('M0+M9',[M[0],M[9]],None),('K8+lin26',[M[8]],[M[2],M[6]]),('K1+lin0',[M[1]],[M[0]])]:\n    r=loo(bl,y,lin=lin); b=max(r.items(),key=lambda t:t[1]); print(name,'LOO best',b,' median',np.median(list(r.values())))\n"
}
```

