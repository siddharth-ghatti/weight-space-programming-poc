## reasoning summary

Layer 1 with an RBF kernel and mean pooling reaches 97.26%, which is surprisingly close to the deeper models. The spread is around ±0.4%, so it's probably within noise. To go further I'm weighing non-uniform attention across the pooling heads, but I don't have much token-level data to work with. I could get it more cheaply from the 'attention' and 'token_hidden' outputs, though those are capped at 64 ids.

I'm wondering whether I can get token-level nonlinear features without reading tokens directly, by building a model whose layers compute features that get mean-pooled. I'd check kernel combinations of the L0 and L1 means too, since the [M0,M1,M2] combo already reached 97.46%. I also want to compare the parameter cost of a two-layer structure against M(1) alone.

I'm weighing a two-layer setup where the second layer is a pooling layer with an unused FFN, which would land around 97.3% in CV, still short of the 97.8% target. I'm also considering a three-layer version that would add the M(2) features on top of M(1).

I'm considering how to choose the pooling heads' query directions without token-level data, perhaps from the class-discriminant weights on the mean features, so the attention can weight tokens in a direction that separates the classes.

I'm weighing whether the pooling heads' query directions can come from the class-discriminant weights on the mean features, so attention concentrates on tokens with strong class-c evidence. The value vectors could carry that per-token score alongside other discriminant directions, which would work like a soft max-pool over token class scores. Evaluating it would take two passes per configuration, or one pass on a 1024-id subset for screening.

I'm checking whether the per-token scale for the head queries can be calibrated from token_hidden on a small batch, since the attention math works on the activations directly. A few passes over roughly 192 sentences would give enough token states to look at which directions are useful.

Now I'm comparing parameter budgets for a few layer configurations. Two layers with a full FFN in layer 0 comes to about 38.6M and reaches roughly 97.3%, while shrinking that FFN to 1024 units brings it to about 32.3M. A three-layer attention-only option at around 31.5M is unclear, and a single pooling layer at 26.8M sits about a point short of the target.

I'm weighing whether the deep path could reach the target with fewer layers, perhaps by keeping only a few BERT layers such as 0, 1, 2, 5 and 9 and skipping the middle ones.

I'm thinking about whether the shallow model could match a deeper one by dropping some BERT layers, though that would need extra evaluation passes. Pruning the FFN in every layer is another cost-saving option, but it gives up a lot of capacity. I think the better bet is improving the shallow features through attention pooling focused on class evidence, plus nonlinear readouts. Before that, I want to look at the misclassified examples from the M1-kernel model to see which confusions, like negation versus coverage, are driving the errors.



Layer-1 features with a kernel readout hit 97.3% CV, which points to a cheap design of layer 0 plus one k=2 pooling layer, but that is still under the 97.8% target. I'm now checking which sentences fail to see what information is missing.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0, str(WORK))\nfrom cvlib import *\nR = np.load(WORK/'base_reads.npz'); y=R['y']; res=np.load(WORK/'reserved.npy'); ids=R['ids']\nkeep = np.setdiff1d(np.arange(768), res)\nM = lambda L: R[f'mean_hidden_L{L}'][:,keep]\nP={x['id']:x for x in practice()}\nX=M(1); f=folds(y,5,0)\nmu=X.mean(0); sd=X.std(0)+1e-6; Z=(X-mu)/sd\nsq=(Z**2).sum(1); D2=(sq[:,None]+sq[None,:]-2*Z@Z.T)/Z.shape[1]; K=np.exp(-0.3*D2); Y=np.eye(8)[y]\nS=np.zeros((len(y),8))\nfor i in range(5):\n    tr=f!=i; te=f==i\n    A=np.linalg.solve(K[np.ix_(tr,tr)]+0.001*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0)); S[te]=K[np.ix_(te,tr)]@A\npred=S.argmax(1); err=np.where(pred!=y)[0]\nprint(len(err))\nfor e in err:\n    x=P[ids[e]]; print(f\"[{LABELS[y[e]][:10]} -> {LABELS[pred[e]][:10]}] ({x['register']},{x.get('variant')}) {x['text'][:160]}\")\n"
}
```

