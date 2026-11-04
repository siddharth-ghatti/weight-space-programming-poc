## reasoning summary

The E1_u.npy file holds the final CLS hidden state of the one-layer edited model, with both passes concatenated into a 1496x768 array, and the FFN is still switched off for now. Next I want to measure accuracy from those linear features before designing the FFN, and I'm reconsidering the strategy, starting with the option where k=0.

I'm weighing whether k=0 could work, but the pooler would only see constant embedding output, so k=1 is the minimal useful choice. Parameter count rises by 1537 per FFN unit, so the real trade-off is accuracy against m, with m=1 as the lower bound. The pooler's tanh layer over 768 units already acts as a free hidden layer, so the open question is whether attention-only features plus that layer can reach the 97.8% target.

I'm thinking about the pooler as a free hidden layer: its 768 tanh units could act as a kernel machine, with anchor-based features and a closed-form ridge fit on the classifier. The FFN would add parameters for little gain, so the pooler carries most of the load. What remains open is how much a single attention layer can pull from the embeddings at the CLS position.

Next I'm checking how the attention heads pool the token embeddings, with the E1 heads meant to pick out class-evidence tokens. Then I'll measure how well E1_u does with a linear ridge fit and with a kernel, against the 96.3% and 97.7% from my notes.

I'm still weighing whether attention alone can reach the target. The original model's 98.3% sits above the 97.8% bar, but a single layer only pools token embeddings at the CLS position, so I'm checking how much contextual signal that can carry.

I'm checking whether two attention layers would pay off: a second layer could mix context between tokens, which one layer can't do, and it costs about 2.4M extra parameters.

I'm now setting up the E1 features for a linear ridge fit and a kernel or random-feature fit. Next I'm looking at the pooler as a fixed tanh layer over the layer-normalized CLS vector, so its 768 units can act as the kernel machine.

I'm weighing whether the pooler's 768 tanh units can serve as the kernel machine, using anchor-based features with a ridge fit on the classifier, which is closed-form and so allowed. I'm also considering mixing in some near-linear units for the ridge scores, and I'm leaning toward skipping the FFN unless the pooler falls short.

I'm going to check the E1 features first, starting with a 10-fold linear CV on E1_u and a kernel-style fit to compare against my earlier numbers. The E1 design isn't saved as a script, so I'll also need to work out what the edited weights are doing before I change anything.



E1_u.npy's final [CLS] features are in hand, and before more forward passes I'll test linear ridge and kernel-style CV on them against the base-model features, since the pooler and classifier can act as a free one-hidden-layer network.



## tool_use: run_python

```json
{
 "code": "exec(open('lib.py').read())\ny=labels()\nU=np.load('E1_u.npy').astype(np.float64)\nprint('U norm', np.linalg.norm(U,axis=1)[:5], U.mean(1)[:3], U.std(1)[:3])\nprint('lin10', cv_acc(U,y,lams=(0.1,1,10,100),k=10))\nB=load_base()\nM0=B['mean_hidden_L0'].astype(np.float64)\nprint('M0 lin10', cv_acc(M0,y,lams=(0.1,1,10,100),k=10))\ndef cosn(X):\n    Xc=X-X.mean(0); return Xc/np.linalg.norm(Xc,axis=1,keepdims=True)\nimport scipy.special as sp\ndef gelu(x): return 0.5*x*(1+sp.erf(x/np.sqrt(2)))\nfor name,X in [('U',U),('M0',M0)]:\n    Z=cosn(X); S=Z@Z.T\n    for sc in [2,4]:\n        K=S+gelu(sc*S)/sc+gelu(-sc*S)/sc\n        print(name, sc, krr_cv(K,y,lams=(0.03,0.1,0.3,1),k=10,seeds=(0,)))\n"
}
```

