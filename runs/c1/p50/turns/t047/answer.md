## reasoning summary

I'm checking the value rows: 0-6 look like class directions, and 8-63 are the PCA components. Using only the 56 PCA dimensions gets about 90.9%, which is weak, so most of the accuracy must be coming from the leaky class directions. The honest accuracy is probably around 96%, but I can't be sure. That means E1 carries little real information, and I need a better design.

I'm weighing information capacity: twelve heads of 64 value dims give 768 dimensions, and a plain mean of embeddings reaches about 96% linearly and roughly 97.7% with a kernel. Carrying the full 768-dim mean would leave no room for attention-weighted features, so I'd have to trade some of that dimensionality against class-evidence pooling. The real question is whether a bag-of-embeddings model can get near the 98.3% reference, which uses twelve contextual layers. A single layer probably tops out around 97.9%, which is a risky margin against the 97.8% threshold given the test noise.

I'm weighing a two-layer design: one contextual layer to compute token representations, then a pooling layer. The cost of keeping the original FFNs is the main concern, since the two-layer version still cuts parameters by about 65% compared with the full model.

I'm checking whether layer 1 features alone, from the original layer 0 with attention kept, could carry enough signal, perhaps by measuring linear and kernel CV on the M1 and M2 hidden states. Pruning the FFN to a smaller width might work better for a single layer than it did across all eleven.

I'm now considering whether layer-A features can be read directly from an edited model, by zeroing layer 0's FFN output so only the attention contribution remains. Then I could take the mean of the token outputs at layer 1 and test that feature without designing the pooling layer first. The first option to evaluate is the original layer 0 with its full FFN at width 3072.

I'm now comparing a few layer-A options: the original layer 0 with its full FFN, attention only with the FFN off, or the FFN pruned to 64 or 256 units. After that I'd add a pooling layer with attention and a kernel head. I'll first check linear and kernel CV on the base layer features M1 and M2, along with the CLS representation.



Dropping the class-direction dims leaves 90.9%, so most of E1's accuracy comes from label-fitted dims and its honest accuracy is probably about 96%. I'm now measuring how separable each base layer's pooled features are, to see whether one or two contextual layers would add enough margin.



## tool_use: run_python

```json
{
 "code": "exec(open('lib.py').read())\nimport scipy.special as sp\ny=labels(); B=load_base()\ndef cosn(X):\n    Xc=X-X.mean(0); return Xc/np.linalg.norm(Xc,axis=1,keepdims=True)\ndef gelu(x): return 0.5*x*(1+sp.erf(x/np.sqrt(2)))\nres={}\nfor l in [0,1,2,3,4,6,12]:\n    X=B[f'mean_hidden_L{l}'].astype(np.float64)\n    lin=cv_acc(X,y,lams=(1,10,100),k=10,std=True)\n    Z=cosn(X); S=Z@Z.T; K=S+gelu(2*S)/2+gelu(-2*S)/2\n    ker=krr_cv(K,y,lams=(0.1,0.3,1),k=10,seeds=(0,))\n    print(l, {k:round(v,4) for k,v in lin.items()}, ker)\n"
}
```

