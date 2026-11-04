## reasoning summary

Naive FFN pruning to 1024 hurts a lot here: deep-layer features end up worse than a plain bag-of-embeddings baseline, and representations drift badly by layer 10. So the damage compounds across layers and undercuts the benefit of depth. That makes narrowing the FFN expensive in accuracy, which points toward keeping the full FFN in a few layers or finding much better pruning or compensation. I'm starting to look at training-free compensation options, beginning with least-squares.

I'm weighing whether a small model can reach the accuracy target by squeezing more out of shallow features, since the depth gain looks like only about 1%. I'll also check the test margin against the CV numbers, because the threshold sits just below the original's test score.

I'm checking whether more training data would help: comparing 5-, 10-, and 20-fold CV accuracy to see if the full-data model should do better than the CV numbers suggest. Then I want to find shallow features that are more useful after mean-pooling, such as token-level nonlinear features from a per-token FFN in an early layer.

I'm thinking about designing the shallow features in closed form rather than training them. The idea is to use random projections with GELU as the FFN in an early layer, so each token gets nonlinear features, and then mean-pool those and fit ridge on top, which should be a richer set-style kernel than mean-then-RBF. I'd need token-level data to design these directly, or I could just pick random features blindly and read the pooled results back out.

I'm considering whether the pooling layer could use attention weights instead of a uniform mean, since that might carry more signal than the plain average of token features.

I'm considering whether the pooling layer could use some attention heads with class-directed queries, so a few heads act like soft max-pooling over class evidence while the rest stay uniform averages. The token-level class directions would come from a ridge fit on the layer-0 output, and I'd read the final CLS features back out and run CV with an RBF head.

I'm working out how to make the attention-based pooling heads read the class evidence: each token's score comes from a ridge direction, the attention weights follow an exponential of that score, and the value and output projections route each head's result into its own block of the 768-dim output. The open question is whether the residual from the CLS token should be kept or zeroed, since it carries useful information from the first layer.

I'm weighing whether the pooling layer could use attention weights that depend on class evidence, with the value and output projections routing each head's result into its own block of the output. The remaining question is how to choose the temperature so the attention neither collapses nor stays uniform, and whether the CLS residual should be kept.

I'm going to check the learning curve first, comparing 5-, 10-, and 20-fold CV, since that tells me how much the full-data model should gain. I'll also rerun the RBF with several seeds on 10-fold so the comparison isn't just noise.



Naive FFN pruning fails: at m=1024 CV drops to 95.7%, worse than bag-of-embeddings. Next I'll check the learning curve across 5-, 10- and 20-fold CV to see how much 5-fold understates the final model.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\nd=load_base(); y=labels()\nM=lambda l: d[f'mean_hidden_L{l}'][:,KEEP]; Cc=lambda l: d[f'cls_hidden_L{l}'][:,KEEP]\nXo=np.concatenate([M(2),M(6),M(10),Cc(12)],1)\nfor k in [5,10,20]:\n    print(k,'orig-lin',cv_acc(Xo,y,lams=(3,10),k=k,seeds=(0,1,2)),\n          'L0 lin',cv_acc(M(0),y,lams=(3,),k=k,seeds=(0,1,2)))\nfor k in [5,10]:\n    print(k,'orig rbf',krr_cv(rbf(blockn([M(2),M(6),M(10),Cc(12)])),y,(0.003,),k=k,seeds=(0,1,2)), '0,1 rbf', krr_cv(rbf(blockn([M(0),M(1)])),y,(0.003,),k=k,seeds=(0,1,2)), 'L0 rbf',krr_cv(rbf(M(0)),y,(0.01,),k=k,seeds=(0,1,2)))"
}
```

