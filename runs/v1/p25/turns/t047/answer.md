## reasoning summary

I'm looking at the first encoder layer's attention, which is attention index 1. For each head there, I want to measure how close the attention rows are to uniform, using the ratio of entropy to log(n) and the average total variation distance from a uniform distribution.



## message

I'll measure how close each layer-0 head (attention_L1 = first encoder layer) is to uniform attention, to pick the cheapest head to sacrifice for M0 pooling.

## tool_use: run_python

```json
{
 "code": "import json, numpy as np\na=np.load('pass_6/attention_L1.npy').astype(np.float64)  # [64,12,T,T]\ntk=json.load(open('pass_6/tokens.json'))\nlens=[len(t) for t in tk]\nres=np.zeros((12,)); ent=np.zeros(12)\nfor i,L in enumerate(lens):\n    A=a[i,:,:L,:L]\n    u=np.ones(L)/L\n    res+=0.5*np.abs(A-u).sum(-1).mean(-1)\n    ent+=(-(A*np.log(A+1e-12)).sum(-1)/np.log(L)).mean(-1)\nprint('TV from uniform per head:', np.round(res/len(lens),3))\nprint('norm entropy per head  :', np.round(ent/len(lens),3))\n# check layer index meaning: sum rows\nprint(a[0,0,:3,:lens[0]].sum(-1))\na2=np.load('pass_6/attention_L2.npy').astype(np.float64)\nres2=np.zeros(12)\nfor i,L in enumerate(lens):\n    A=a2[i,:,:L,:L]; res2+=0.5*np.abs(A-1/L).sum(-1).mean(-1)\nprint('L2 TV:', np.round(res2/len(lens),3))"
}
```

