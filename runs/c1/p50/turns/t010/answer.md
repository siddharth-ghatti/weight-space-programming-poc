## reasoning summary

The layer-0 bag-of-embeddings baseline already reaches about 95.7%, and the deeper layers add only a little, reaching 96.4% at best, so this task seems mostly lexical. The deep model isn't contributing much on its own. I'm weighing whether smaller lambdas, standardized features, or combining layer 0 with layer 12 would help more.

I'm going to try a few more feature combinations, like layers 0 with 1, 6, or 12, since a single layer on its own only gets about 95.7% and I need closer to 98%. I'm also thinking about what one layer could actually compute for the CLS token, starting with attention, where the CLS query is effectively constant and each head becomes a weighted average over the token values.

I'm working out how a single layer could approximate the deeper model: each of the 12 attention heads would act as a pooling step over the tokens, with the CLS query fixed, so each head is a weighted average shaped by a learned direction. That gives 12 different pooling patterns, each contributing 64 dimensions that get combined through the output projection.

I'm now considering whether the attention heads could each act as a separate pooling step, with each head carrying a small set of discriminative projections, like eight ridge scores per head. The open question is how to choose each head's attention direction without training, and one idea is to point it at the tokens most indicative of each class, so the softmax behaves like a soft max over class evidence.

I'm checking whether a single attention layer could approximate the pooling I need: since the CLS query is constant, each head becomes a weighted average over tokens whose weights come from a fixed direction. I'm considering setting that query so the softmax picks out tokens strongly tied to each class, which would act like a soft max over class evidence alongside the plain mean.

Setting the query so each head scores tokens by a fixed direction, the CLS token's pooled output then runs through the residual norm, the feed-forward block, and a final norm.

I'm now looking at how the LayerNorm after attention would rescale things. Since its division by standard deviation varies per sentence, I'd want a large constant component so the scale stays roughly fixed and the features pass through approximately linearly. The float precision should be fine, and the LayerNorm gain can rescale the chosen dimensions afterward.

I'm now checking whether the FFN can help: a few hundred units would cost only about 150K parameters, which is small next to a full layer, so I could use them as cheap nonlinear features if they help reach the target.

I'm going over the parameter budget for a one-layer or two-layer model, which comes to roughly 26.8M versus the original 109.5M. The harder part is whether such a small model can reach about 98% when the mean-embedding baseline sits near 95.7%. I want to see whether token-level scoring and pooling can close that gap, but that means working out the embedding output in numpy first.

I'm checking whether computing the token embeddings myself from the weights would count as reimplementing the forward pass, since the sandbox rules forbid that. Getting the layer-0 hidden states from the real forward pass would take about 24 batched reads, which is expensive, so I'm weighing that cost against the analysis I'd gain.

I'm settling on a design where the edited model's CLS output carries the pooled features I want, so one forward pass gives me them and I can fit ridge offline. Each experiment costs about two reads of the 1,496 sentences, so I'll have to pick the attention-pooling designs carefully rather than trying many.

I'm going to check how the mean-pooled layer features do on their own, including the smaller-lambda and standardized variants, then try layer 0 paired with layers 1, 6, and 12. After that I'll look at whether the CLS and SEP vectors carry length information in the mean, since they scale with the token count.

I'm thinking about whether a random-feature layer, with 768 tanh units applied to the mean features before a ridge fit, could beat the linear version, though I expect it won't do much better for this data. I'd check that offline against the pooled outputs from a forward pass rather than computing the pooler myself.

Now I'm just going to run the combinations.



Plain averaging of the embedding-layer outputs already reaches 95.7% under ridge CV, and no deeper layer goes past about 96.4%, so the task is mostly lexical. Next I'll test whether combining layers helps and tune the ridge penalty more finely.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\nd=load_base(); y=labels()\nM=lambda l: d[f'mean_hidden_L{l}'][:,KEEP]\nC=lambda l: d[f'cls_hidden_L{l}'][:,KEEP]\nr=cv_acc(M(0),y,lams=(0.1,0.3,1,3,10,30)); print('L0', {k:round(v,4) for k,v in r.items()})\nr=cv_acc(M(0),y,lams=(30,100,300,1000,3000),std=True); print('L0 std', {k:round(v,4) for k,v in r.items()})\nfor combo in [(0,1),(0,2),(0,6),(0,12),(1,6),(0,6,12),(2,6,10),(0,3,6,9,12),(0,1,2,3)]:\n    X=np.concatenate([M(l) for l in combo],1)\n    r=cv_acc(X,y,lams=(3,10,30,100)); print(combo, {k:round(v,4) for k,v in r.items()})\nX=np.concatenate([M(0),C(12)],1); print('L0+cls12', cv_acc(X,y,lams=(3,10,30,100)))\nX=np.concatenate([M(2),M(6),M(10),C(12)],1); print('orig-like', cv_acc(X,y,lams=(3,10,30,100,300)))"
}
```

