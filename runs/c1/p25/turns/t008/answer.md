## reasoning summary

Layer 0 alone, the embedding mean, already reaches about 96.0%, barely behind the later layers at roughly 96.1 to 96.5%, so depth adds little for a single mean-pooled layer. Combining layers pushes it to about 97.8%, so next I'm testing pairs and triples of layers to see which combination works best. I also need to remember that the ridge regression here standardizes its features, while the model's pooler can absorb that standardization through its linear layer on the final CLS token.

I'm also considering whether the LayerNorm on the final CLS token changes how the standardization would play out inside the model, and I want to test mean features that include the CLS token from layers 1 and 2 as well. I'll evaluate those combinations next.



Pooled embeddings alone already hit about 96% CV, and single deeper layers add little, so depth seems to help mainly in combination. I'm now searching for small layer combinations that reach roughly 97.8%.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0, str(WORK))\nfrom cvlib import *\nR = np.load(WORK/'base_reads.npz'); y=R['y']; res=np.load(WORK/'reserved.npy')\nkeep = np.setdiff1d(np.arange(768), res)\nM = lambda L: R[f'mean_hidden_L{L}'][:,keep]\nC = lambda L: R[f'cls_hidden_L{L}'][:,keep]\ncombos = {'0+1':[M(0),M(1)], '0+2':[M(0),M(2)], '1+2':[M(1),M(2)], '0+1+2':[M(0),M(1),M(2)], '0+3':[M(0),M(3)],'0+2+3':[M(0),M(2),M(3)],\n          '0+1+c1':[M(0),M(1),C(1)], '0+2+c2':[M(0),M(2),C(2)], '0+6':[M(0),M(6)], '2+6+10+12':[M(2),M(6),M(10),M(11)], '0+1+2+3+4':[M(0),M(1),M(2),M(3),M(4)]}\nfor k,v in combos.items():\n    X=np.concatenate(v,1); a=cv_acc(X,y,lams=(100,300,1000,3000))\n    print(k, {kk:round(vv,4) for kk,vv in a.items()})\n"
}
```

