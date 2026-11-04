## reasoning summary

The 2+6+10+11 combination hits 97.8%, but shallow combinations top out around 96.4%, so reaching 97.8% seems to require deep features, which would mean something like 12 layers. I'm wondering whether nonlinear readouts or attention pooling could lift the shallow features instead. The embedding mean is already at 96%, so the remaining errors are probably on harder sentences that need context.

I'm checking whether the test score of 98.31% means the test set is slightly easier than my CV of about 97.5 to 97.8%, or whether the gap is just noise. Either way, reaching the test threshold feels risky. I'm also weighing the cost of 12 layers with very small FFN widths, which I'd guess degrades the features badly, against fewer layers with full FFN.

I'm now seeing that a joint linear ridge over the means of several layers can be folded into the 8 reserved dimensions, since each layer's carry head contributes its share of the same linear score. That would let a shallow structure use features from layers 0 through j plus the pooling layer. The catch is that each carry head takes over one of the 12 heads, which may slightly damage that layer's own computation.

I'm weighing whether to carry the embedding mean through layer 0's heads, since that layer's input is the raw embedding hidden state. Then I'd compare joint ridge fits over the means of hs[0] through hs[j+1] for each j from 0 to 11, and check whether adding the CLS vector at the end helps.



Combined layers reach about 97.8% against roughly 96.4% for shallow ones, and I'm now measuring a joint ridge over the prefix layers 0..j to see which feature set those early layers can offer.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0, str(WORK))\nfrom cvlib import *\nR = np.load(WORK/'base_reads.npz'); y=R['y']; res=np.load(WORK/'reserved.npy')\nkeep = np.setdiff1d(np.arange(768), res)\nM = lambda L: R[f'mean_hidden_L{L}'][:,keep]\nfor j in range(1,13):\n    X=np.concatenate([M(i) for i in range(j+1)],1)\n    a=cv_acc(X,y,lams=(300,1000,3000,10000))\n    X2=np.concatenate([M(i) for i in sorted(set([0, j//2, j]))],1)\n    b=cv_acc(X2,y,lams=(100,300,1000,3000))\n    print(j, 'all', {kk:round(vv,4) for kk,vv in a.items()}, '0,mid,j', {kk:round(vv,4) for kk,vv in b.items()})\n"
}
```

