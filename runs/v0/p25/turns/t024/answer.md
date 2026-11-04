## reasoning summary

I'm estimating that the perturbation to q_i is only about 2-4% of its norm, so it's small. Next I'm applying the layer-0 edit across all 12 heads: since the stacked query weights take a rank-1 update, W_q' becomes W_q minus the outer product of q_CLS and r, with the bias left unchanged. I'm about to check what q_CLS' comes out to.

Checking that the rank-1 edit actually zeroes out q_CLS' came out clean. Next I'll read the CLS and mean hidden states across all layers for the 1496 examples, then run the cross-validation on the edited CLS and mean features.



Perturbation to other tokens is about 2–4% of their norm, with zero change for [SEP] and '.'. Next I'll apply the layer-0 rank-1 query edit across all 12 heads and read the cls and mean hidden states at every layer for the practice sentences.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nr=np.load('r0.npy'); D=np.load('base_feats.npz'); eC=D['cls0'][0].astype(np.float64)\nWq=base('bert.encoder.layer.0.attention.self.query.weight').astype(np.float64); bq=base('bert.encoder.layer.0.attention.self.query.bias').astype(np.float64)\nqC=Wq@eC+bq\nWq2=Wq-np.outer(qC,r)\nprint('new q_CLS norm', np.linalg.norm(Wq2@eC+bq))\nsave_edit('bert.encoder.layer.0.attention.self.query.weight', Wq2.astype(np.float32))\nwith open('notes.md','a') as f: f.write(\"\"\"- run21: layer0 CLS-uniform trick: r0.npy (r.e_CLS=1, r.x~0 others, exact 0 for SEP/'.' and pos1..160), \n  edit L0 query.weight = Wq - outer(q_CLS, r) -> CLS query exactly 0 in all heads (uniform attn, absorbs mean0), others perturbed ~2%.\n  Next: read cls/mean all layers (passes 6,7) to see if cls11 now carries bow info.\n\"\"\")"
}
```

