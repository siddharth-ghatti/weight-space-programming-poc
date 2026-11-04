## reasoning summary

I'm comparing the stacked model at about 0.980 against 0.975, roughly a half-percent gain, though the injected signal may get distorted passing through the early layers, which also perturbs BERT's other tokens. I'm wondering whether the bow information could instead go in without that transit, looking at how the pooled output is computed from the CLS hidden state.

I'm checking whether the bow signal could bypass the layer stack entirely, since the pooled output only reads the CLS state at layer 12, which itself depends on every token's layer-11 hidden state. That means the information has to travel through layers 1 to 10 in some token, and I'm weighing whether to carry it in CLS alone or spread redundant copies across all tokens, though either route still runs into the same distortion problem.

I'm weighing whether the injected signal can be made honest despite the supervised injection, since the final classifier would otherwise overweight the leaked bow scores. The obvious fixes are a label-free projection like PCA, or hand-picked keyword directions in embedding space, though the fixed injection direction still has to survive the transit through the layers.

I'm considering a label-free alternative: a head that projects each token onto a set of hand-picked keyword embedding directions, so the downstream fit stays honest. Dot products with unrelated words aren't zero, so that signal would be noisy. Another option is strongly shrunk supervised directions, though the combiner would still lean on them a bit too much.

I'm checking whether the train-versus-CV gap for the ridge fit on the m0 features is small, since that would mean the leakage from the injected signal is minor. I'm also asking how the CLS processing will transform the injected signal, and whether a large-norm injection would push the CLS state out of distribution for layers 1 through 10.

I'm weighing a way to keep CLS from disturbing the other tokens in layers 1 to 10, perhaps by writing a flag into CLS at layer 0 that makes its keys nearly invisible to queries, though that flag would itself have to be carried through the later layers.

I'm working out whether a constant flag can ride along in CLS's value at layer 0, and the answer looks like no: for non-CLS queries the CLS value is W_V e_CLS + b_V, which I'd have to set to zero, so CLS's output is just W_V(mean0 - e_CLS) with no constant offset to exploit.

I'm weighing whether a constant flag could be carried in CLS's value, but it would only come out as roughly 1/n and vary with length, so that route is out. Before going further down the injection path, I want to check whether simply using cls11, mean11 and cls12 together gives a cheaper gain of similar size.

I'm checking whether using cls11, mean11 and cls12 together gives a cheaper gain of similar size, since the layer-11 CLS output is already computed in the original model and needs no injection.



Nested CV suggests a supervised bag-of-words summary could raise accuracy from about 97.5% to 98.0%. Before the invasive layer-0 changes, I'm checking whether cheaper layer-11 signals, such as cls12, add anything.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nD=np.load('base_feats.npz'); y=D['y']; folds=np.load('folds.npy')\nF=lambda k: D[k].astype(np.float64)\nfor name,X in [('c11+m11',np.hstack([F('cls11'),F('mean11')])),\n               ('c11+m11+c12',np.hstack([F('cls11'),F('mean11'),F('cls12')])),\n               ('c11+m11+m10',np.hstack([F('cls11'),F('mean11'),F('mean10')])),\n               ('c11+m11+m6',np.hstack([F('cls11'),F('mean11'),F('mean6')])),\n               ('c11+m11+m2',np.hstack([F('cls11'),F('mean11'),F('mean2')])),\n               ('c11+m11+m1',np.hstack([F('cls11'),F('mean11'),F('mean1')])),\n               ('c11+m11+m0',np.hstack([F('cls11'),F('mean11'),F('mean0')]))]:\n    print(name,[round(cv(X,y,folds,ridge_fit,l)[0],4) for l in [100,300,1000]],flush=True)\n# train vs cv gap of bow ridge\nfor lam in [100,300,1000,3000]:\n    Wt,b=ridge_fit(F('mean0'),y,lam); tr=(np.argmax(F('mean0')@Wt.T+b,1)==y).mean()\n    print('bow ridge',lam,'train',round(tr,4),'cv',round(cv(F('mean0'),y,folds,ridge_fit,lam)[0],4))"
}
```

