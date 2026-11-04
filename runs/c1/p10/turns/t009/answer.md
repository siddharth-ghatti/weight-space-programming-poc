## reasoning summary

Layer 0 averaging, which is just embeddings, already reaches 96.2%, and layer 9 gets 97.2%, but nothing hits 98% on its own. The editor's combination sits around 97.8-97.9%. That points to a task that leans heavily on specific keywords, so I'm wondering whether a k=1 model's attention could do much better than uniform averaging.

I'm weighing whether the test set is a little easier than cross-validation suggests, since the base model hit 98.31% on test against 97.8% CV, so I want CV at least around 97.8%. For improving the features, I could pool several layers' means, add a nonlinear classifier over the features, or use FFN-computed nonlinear token features before mean-pooling.

I'm now checking whether the attention layer could beat uniform averaging, with a single layer of 12 heads pulling in token information from the CLS position. The rough idea is that the output projection plus residual could carry the full mean embedding, and I'm weighing how much the layer norm and the FFN change that picture.

The FFN and pooler could add nonlinearity on top of the CLS features, perhaps with random features and ridge for a kernel classifier. I doubt that alone gets to 98%: a bag-of-embeddings model is limited, though it might reach around 96.5-97%. Attention with non-uniform weights might do better by emphasizing specific keywords.

Now I'm imagining each head's key tuned to a per-token class score, so the softmax weighting pulls out the most class-relevant tokens and acts like max-pooling of keyword evidence.

I'm thinking about whether the token-level features could be computed directly from the frozen embedding weights, since reading hidden states through the forward pass is capped at 64 ids per call, which is a real constraint for fitting anything per-token. Embedding lookup plus layer norm is standard math on fixed weights, but I'm unsure whether doing it by hand counts as reimplementing the forward pass in a way I'm supposed to avoid.

I'm reconsidering whether computing the layer-0 embedding output by hand, from the frozen word and position weights, is allowed. Reading token hidden states through the forward pass is capped at 64 ids per call, which makes fitting per-token features awkward. Using the weights directly as closed-form math feels like it might cross into reimplementing the forward pass, so I'm leaning toward avoiding it for evaluation.

Now I'm comparing a safer route that keeps the early layers and pools a middle or later layer, with the full FFN in place, against a much smaller model that leans hard on nonlinearity but risks underfitting. Before choosing, I want the cross-validated scores for the layer combinations, since that will show how close the prefix can get to the editor's 97.8-97.9%.

I'll check the combinations first, and also test whether kernel or random features on the layer-0 means could take a k=1 model further.



The embedding output alone already reaches 96.2% CV, and no single layer goes beyond about 97.2%. Next I'll test layer combinations to see how deep the model must be to reach about 98%.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nfrom lib import *\nF = np.load(WORK/'base_feats.npz'); y=F['y']\nM = {L: F[f'mean_hidden_L{L}'][:,KEEP] for L in range(13)}\nC = {L: F[f'cls_hidden_L{L}'][:,KEEP] for L in range(13)}\ndef ev(blocks, lams=(100,300,1000,3000)):\n    X = np.concatenate(blocks,1)\n    return [round(cv_acc(X,y,l),4) for l in lams]\ncombos = {\n 'M0+M1':[M[0],M[1]], 'M0+M2':[M[0],M[2]], 'M0+M3':[M[0],M[3]], 'M0+M4':[M[0],M[4]],'M0+M5':[M[0],M[5]],\n 'M0+M6':[M[0],M[6]], 'M0+M9':[M[0],M[9]], 'M1+M5+M9':[M[1],M[5],M[9]], 'M1+M5+M9+C12':[M[1],M[5],M[9],C[12]],\n 'M0+M2+M4':[M[0],M[2],M[4]], 'M0+M1+M2+M3':[M[0],M[1],M[2],M[3]],\n}\nfor k,v in combos.items(): print(k, ev(v))\n"
}
```

