## reasoning summary

I'm checking whether pass_41's files line up with the exposed layers, since token_hidden_L7 would be the output of the seventh layer, which is the exposed one. The L8 file doesn't fit a [7] layer list, so I'm wondering whether layer index 8 just isn't part of this structure.

I'm going to check the expo_map.json file to see whether the L8 files were renamed to match the original layer indices. Then I'll compare the fits for layer 7 against target 8, for a skip to target 9, and for the baseline gap between h8 and h9. For the L5 comparison I'll look for the pass that holds that exposure.



Pass_41 is the L7 exposure; I'm confirming the pass-to-layer mapping in expo_map.json, then comparing held-out errors for a normal refit versus a skip-refit.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nimport json\nprint(json.load(open('expo_map.json')))\nfrom lib8 import *\nimport numpy as np\nX,sid,msk=get_X(7,41)\nnS=msk.shape[0]; rng=np.random.RandomState(0); tr=np.ones(nS,bool); tr[rng.choice(nS,16,replace=False)]=False\nfor (tgt,lnf) in [(8,7),(9,8)]:\n    keep,Wf,bn,info,Hb,X,sid,msk=fit_skip(7,41,tgt,1024,W,base,lam=100.0,tr=tr,ln_from=lnf)\n    print('L7 ->h',tgt,info)\nH8=np.load('pass_20/token_hidden_L8.npy')[msk].astype(np.float64); H9=np.load('pass_20/token_hidden_L9.npy')[msk].astype(np.float64)\nd=H8[:,KEEP]-H9[:,KEEP]; print('h8 vs h9 rel err', (d**2).sum(1).mean()/(H9[:,KEEP]**2).sum(1).mean())\nH7=np.load('pass_20/token_hidden_L7.npy')[msk].astype(np.float64)\nd=H7[:,KEEP]-H8[:,KEEP]; print('h7 vs h8 rel err', (d**2).sum(1).mean()/(H8[:,KEEP]**2).sum(1).mean())"
}
```

