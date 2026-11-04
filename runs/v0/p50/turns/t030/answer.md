## reasoning summary

The multi-layer result comes out around 98.5% against 98.0% for the single layer, a gain of about half a percent, or seven items out of 1,496. Mean-pooling layer 9 alone already reaches 98.2%. Combining mean0 with mean11 also gives 98.5%, and layer 0's mean is just the mean of the embeddings output after the frozen embeddings and layer norm. I'm wondering whether mean0 could go into the final CLS, and whether there's a way to pool at an earlier layer L for CLS only.

I'm considering a two-route idea: splitting layer 11's twelve heads so some pool h11 while others handle something else, though they all see the same input. Another option is to pool at an earlier layer such as 9 and run layers 10 and 11 on the CLS token alone, which would give a deeper kernel but drop mean11. I'm also wondering whether keeping other tokens unchanged while pooling at layer L is possible, since all heads are already in use.

I'm considering repurposing one of layer 8's less important heads to attend uniformly, so every token receives the same pooled vector. The change would ripple through layers 9 through 11, and since the classifier is refit on the new features, what matters is whether the information survives. It's unclear whether this helps, and each test would take two passes, with 43 left to try.

I'm considering whether a constant vector added to every token at layer L would survive into mean-pooled layer 11. Since uniform attention gives every token the same output, the shift is identical across tokens, so it shouldn't change the attention pattern through the keys, though it would still affect the queries and values in later layers.

I'm weighing whether to change the attention of a layer so every token gets the same pooled vector, since a constant shift would pass through the keys unchanged. The gain is only about half a percent, but the 43 remaining passes give room to test it, and CV would catch any damage from the layer changes. I'm also thinking about whether the residual stream keeps enough token identity from the embeddings that mean0 could be folded into the final representation.

I'm thinking about whether attention heads could focus on specific keyword tokens instead of pooling uniformly, which might add signal beyond the mean. For the remaining work, using all 1496 centers in the final fit is an automatic gain, and tuning or ensembling looks like a plateau. I'll lock in the current best setup on all the data as a checkpoint before trying the constant-shift idea.

Next I'm locking in the best setup, fitting on all 1496 centers with lam=0.5, and then running a prediction pass over the 512 practice items to confirm training accuracy is close to 100%.



Multi-layer means would add only about 0.5% and carry risk, so I'm checkpointing the safe kernel machine trained on all 1496 practice items, then saving it and noting it.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nimport build_ffn as B\nlab=np.load('feats/y.npy'); E=np.load('feats/edA_cls12.npy')\ntr=np.ones(len(lab),bool)\np=B.fit_kernel(E,lab,tr,a=2.0,t=0.0,lam=0.5,rho=1.0)\nprint('train acc', (p['scores'].argmax(1)==lab).mean())\nnp.save('feats/full_numpy_scores.npy',p['scores'])\ned=B.build_layer11_ffn(p)\nos.makedirs('ckpt_B',exist_ok=True) if False else None\nimport os; os.makedirs('ckpt_B',exist_ok=True)\nfor k,v in ed.items(): save_edit(k,v)\nfor k in edits(): np.save(f'ckpt_B/{k}.npy', W(k))\nprint(len(edits()))\nwith open('notes.md','a') as f: f.write(\"- CHECKPOINT B saved (run 25): kernel FFN fitted on ALL 1496 centers, a=2,t=0,lam=.5,rho=1. Copies of all edits in ckpt_B/. Expected test ~98%.\\n- Kernel CV on multi-layer means: 8+11 or 0+11 ~98.5% vs 11 alone ~98.0% (possible +0.5% if realizable).\\n\")\n"
}
```

