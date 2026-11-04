## reasoning summary

The tanh anchor head lands at about 96.9%, roughly half a point under the RBF kernel ridge baseline at 97.39%. I'm considering whether combining linear features with the anchors could close that gap, or whether fewer anchors with a better kernel shape would do it. The saturation of tanh at high similarity might be what's limiting it, and I'm weighing whether a more negative b with a smaller a, with (2, -1) so far the best setting, could help.

I'm weighing whether the FFN's GELU layer could serve as the nonlinear anchor-feature layer, with the output projection trained as a ridge readout onto the class scores. That would make the FFN behave like a closed-form kernel machine, though I'd need to compute the GELU features offline from the pooling layer's CLS activations to fit it.

I'm checking whether the FFN's GELU layer could act as the nonlinear anchor-feature layer, but the cleaner route is to keep the pooler as the nonlinearity: set it to the anchors, read the actual pooled outputs from the model, and fit the classifier ridge on those.

Testing a 768-anchor tanh head for the k=1 setup, since L0 features plus the head land near 96.9%, still short of the 98%+ target. Attention pooling might add enough signal, so I'm looking at k=2 by pairing the layer-0 features with the pooling layer's output and checking the anchor-head numbers for each combination.

I'm comparing the k=2 options now, checking the anchor-head results for the layer-0 and pooling-layer features alone and in combination, at 10 and 20 folds. If the combined RBF reaches about 98% at 20 folds, a two-layer setup could be workable. I'm also weighing the size tradeoff: a two-layer model at m=3072 is around 35% of the original, while one layer is about 24.5%.

I'm checking whether the pooling layer's 3072-unit FFN could serve as the nonlinear feature layer, but I can't read its activations directly, so I'd need the FFN input reconstructed from the layer's outputs first.

I'm checking whether the FFN output projection could be zeroed and the LayerNorm parameters set so the final hidden state comes out as the intermediate activations, which would let me read them directly. That still leaves the FFN's GELU features to compute in numpy, which feels like simulating the forward pass, so I'm not sure it's allowed.

I'm weighing whether to skip the FFN as a nonlinear layer, since fitting its GELU features would mean simulating part of the forward pass in numpy. The cleaner route is to keep the pooler as the only nonlinearity and fit the classifier on the pooled outputs read from the model. Next I'll test two-layer candidates offline, using the pooling layer's CLS features and the layer-0 mean, to see if the combination gets closer to 98%.



The tanh anchor head reaches about 96.9%, close to the 97.4% of exact RBF kernel ridge but still short of the roughly 98% a 1-layer model needs. Next I'll test 2-layer features made from layer 0 plus a pooling layer, scored at 10 and 20 folds with RBF and anchor heads.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\nd=load_base(); y=labels()\nM=lambda l: d[f'mean_hidden_L{l}'][:,KEEP]; Cc=lambda l: d[f'cls_hidden_L{l}'][:,KEEP]\nfor name,bl in [('M1',[M(1)]),('M0M1',[M(0),M(1)]),('M1C1',[M(1),Cc(1)]),('M0M1C1',[M(0),M(1),Cc(1)]),('M2',[M(2)]),('M0M1M2',[M(0),M(1),M(2)]),('M0M2C2',[M(0),M(2),Cc(2)])]:\n    X=blockn(bl)\n    print(name,'lin10',max(cv_acc(X,y,lams=(0.1,0.3,1),k=10).values()),'rbf10',krr_cv(rbf(X),y,(0.003,),k=10,seeds=(0,1)),'rbf20',krr_cv(rbf(X),y,(0.003,),k=20,seeds=(0,)))"
}
```

