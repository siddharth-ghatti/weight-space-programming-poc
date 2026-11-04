## reasoning summary

The pivoted Cholesky selection at 1000 anchors gives 98.55, close to the all-anchors result of 98.51, so the kernel part of m=1024 looks fine. The open question is how much accuracy pruning costs at 1024 versus 1536, and on the 1024 subset the exact KRR numbers are 97.81 for K9e+lin0 at m=1024, 98.09 at m=1536, and 97.95 for the base product kernel.

The base K9+lin0 number on the 1024 subset is the missing piece: if it comes out near 98.1, the pruning costs about 0.3 points at m=1024 and essentially nothing at m=1536. Projected to full-data CV, m=1024 lands around 98.15 and m=1536 around 98.45. Given an unknown test size, m=1024 carries a noticeably higher chance of dipping under the 97.8 threshold, and drift-corrected sequential pruning might claw back part of that gap.

I'm working out the budget for the final build: exposing the pool layer to read the FFN input at CLS costs two passes over the 1496 ids, installing the FFN kernel and readout plus verification takes about two or three more, and there's roughly five passes of margin for fixes. That leaves about 18 passes, enough for drift-corrected pruning.

I'm sequencing the rest of the build: first the layer-0 carry and cuts with provisional weights, then drift-corrected pruning at m=1024 across layers 0 through 8, then the pool layer read and the final fit and verification. That comes to roughly 14 passes, which fits within the 28 I have left, though the carry weights will still have to be refit once the final features are known.

I'm checking whether the carried values stay small enough to leave the layer-norm statistics essentially unchanged across the nine layers, since each norm divides them by the per-token scale. My plan is to install the carry head with a provisional projection and refit its weights at the end, so the rest of the network shouldn't shift much.

I'm checking how much the carried value shrinks through the layer norms: each one divides it by the token's standard deviation, which would be around 1.5 and would drag it down by a factor near 1500 over eighteen normalizations unless the residual gain offsets that. The per-token spread of those divisions makes the carried value noisy, but the ridge fit works from the measured values, so the average is what counts.

I'm checking how much the carried value varies from sentence to sentence, since each layer's normalization scales it by a token-dependent factor. If those factors stay fairly consistent, the carry term is roughly a constant multiple of what the fit assumed. Otherwise I'd need to measure that multiplier per sentence before deciding whether to keep the current carry design.

I'm going to compare the carried scores still sitting in the reserved dimensions at layer 9 against what the previous editor's layer-2 and layer-6 heads should have produced from the base mean hidden states. Working out which head carries the layer-2 score means checking which query-bias entries changed, and then computing the expected value linearly from the base layer-2 mean. The gap between expected and measured tells me how much the layer-norm divisions vary from sentence to sentence.

I'm checking how much the carried value shifts as it passes through the later layers: the measured per-layer means of the reserved dimensions should show whether the layer-2 score survives into layers 3 through 6 at a steady scale. Comparing those readings against what the layer-2 head would predict from the base mean state will show how much the normalization divisions vary across sentences.

I'm comparing the reserved-dimension means at layer 3 against those at layer 6, since the ratio between them should show how much the layer-norm scaling over layers 3 through 5 varies from sentence to sentence. If that ratio stays close to constant across samples, the carried value's scale is stable enough for the fit to hold.



Pivoted-Cholesky anchors match the full kernel at 98.55% with 1000 anchors, so m=1024 is enough. Next I'll check whether the LayerNorm rescaling of reserved-dim values is stable enough across sentences, using the layer-2 carry.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0, str(WORK))\nfrom lib import *\nd=np.load('base_feats.npz')\nR={l:d[f'mean_hidden_L{l}'][:,RES].astype(np.float64) for l in range(0,13)}\nfor l in [2,3,4,5,6,7,9]:\n    print(l, 'mean|res|=%.4f'%np.abs(R[l]).mean(), 'centered std per dim', np.round(R[l].std(0),4)[:4])\n# ratio of centered-over-dims values L6/L3\ndef cen8(r): return r-r.mean(1,keepdims=True)\na=cen8(R[3]); b=cen8(R[6])\nrho=(a*b).sum(1)/(a*a).sum(1)\nresid=np.linalg.norm(b-rho[:,None]*a,axis=1)/np.linalg.norm(b,axis=1)\nprint('rho L6/L3: mean %.4f std %.4f cv %.3f; fit resid %.3f'%(rho.mean(),rho.std(),rho.std()/rho.mean(),resid.mean()))\n# per-layer gammas for reserved dims\nfor l in range(0,9):\n    g=W(f'bert.encoder.layer.{l}.output.LayerNorm.weight')[RES]; ga=W(f'bert.encoder.layer.{l}.attention.output.LayerNorm.weight')[RES]\n    print(l,'attnLN g',np.round(ga[:4],3),'outLN g',np.round(g[:4],3), 'outLN b', np.round(W(f'bert.encoder.layer.{l}.output.LayerNorm.bias')[RES][:4],3))"
}
```

