## reasoning summary

I'm comparing three configurations: the 1536 run agrees at 0.977 with 23 flips and a low-end std of 0.55, the 1792 run is at 0.981 with 19 flips, and the 2048 run at 0.985 with 15 flips. The low-end deltas shrink as the size goes up, from 0.55 to 0.43 to 0.39, so I'm now estimating how many flips a 400-item test would produce from that spread.

I'm working out the flip counts for each configuration, which come to about 17.6, 13.8, and 12.6 for the three sizes. From there I'm estimating failure probability on the 400-item test: roughly 31% at 1536, 24% at 1792, and 21% at 2048. Those are crude figures, and a plausible change to the per-flip assumptions would shift them, so I'm keeping in mind that any choice here is a gamble at around the 20 to 30 percent level.

I'm now computing the sigma of the logit differences for the new 1536 data. If the lower-layer errors are about 12% smaller, that should show how sensitive the output is to upstream error. I'm also estimating what the 2048 setup would look like with the new data, using the scaling trend. A cleaner way to do this is to find the dominant error directions in the upstream layers, since only a few directions actually move the final logits. Without gradients, I'd fit a regression from the per-item mean and CLS hidden errors at each layer to the final logit deltas and see which directions carry the effect.

I'm now thinking about a selection-based approach instead: since a metric only reweights output dimensions and each dimension's least-squares fit is already optimal on its own, the real levers are which hidden units get picked and how tokens are weighted, with selection guided by the sensitive error directions.

I'm weighing whether to try the upstream route at all: sensitivity might be concentrated in a few directions, but the attention mixing makes that hard to pin down, and a 1000-item linear regression to four logit deltas would be noisy. That's a research project with uncertain payoff, and I'm about 95 turns in with a handoff at 120. I'm leaning toward a cheaper alternative, the L5 fit.

I'm checking whether the layer-2 and layer-3 errors changed the outcome, comparing the earlier pass against the newer one.



Honest m=1536 agrees 0.977 with its reference, with 23 flips and perturbation spread 0.37 (0.55 for low-margin items), versus 0.981/19 flips for m=1792 and 0.985/15 flips for the old m=2048. Next I'll test sensitivity to upstream quality by comparing pass 13's and pass 20's L0–2 fits with the unpruned L5 FFN.



## tool_use: run_python

```json
{
 "code": "\nimport sys; sys.path.insert(0,str(WORK))\nimport numpy as np, json\nfrom lsbuild import gelu, layer_data\nfrom evalh import Lb, lab1, f1\ndef LN(x,g,b): mu=x.mean(1,keepdims=True); v=x.var(1,keepdims=True); return (x-mu)/np.sqrt(v+1e-5)*g+b\npre='roberta.encoder.layer.5.'\nab,W1,b1,U,b2=layer_data(5,base)\ng2=base(pre+'output.LayerNorm.weight').astype(np.float64); bb=base(pre+'output.LayerNorm.bias').astype(np.float64)\nD=base('classifier.dense.weight').astype(np.float64); bd=base('classifier.dense.bias'); O=base('classifier.out_proj.weight'); bo=base('classifier.out_proj.bias')\nhead=lambda h: np.tanh(h@D.T+bd)@O.T+bo\npb=Lb.argmax(1); srt=np.sort(Lb,1); marg=srt[:,-1]-srt[:,-2]; i2=np.argsort(Lb,1)[:,-2]\nhb5=np.load('pass_1/cls_hidden_L5.npy')\nfor f,pc in [('a5e_1536.npy',13),('a5e_1536b.npy',20),('a5e_1792.npy',19)]:\n    ae=np.load(f).astype(np.float64); Ge=gelu(ae@W1.T+b1); L=head(LN(ae+Ge@U.T+b2,g2,bb))\n    pe=L.argmax(1); fl=pe!=pb; dm=(L[np.arange(1000),pb]-L[np.arange(1000),i2])-marg\n    e5=np.sqrt(((ae-ab)**2).sum(1).mean())\n    h5=np.load(f'pass_{pc}/cls_hidden_L5.npy'); \n    print(f,'flips',fl.sum(),'dm std %.3f low %.3f'%(dm.std(),dm[marg<1].std()),' a5 cls rms err %.3f'%e5, ' h5 cls err %.3f'%np.sqrt(((h5-hb5)**2).sum(1).mean()))\n"
}
```

