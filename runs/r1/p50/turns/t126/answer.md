## reasoning summary

I'm comparing v3 against the others on the stress test, where it does worse than expected, with 59 versus 48 flips at k=2. The gap of about 11 is suggestive but not very significant, and I'm wondering whether v3's noise lands on low-margin items more often. I'm also checking whether lg_m1536_v2.npy really matches pass25, and now I'm turning to pass12.

I'm now going through the remaining checkpoints, pass12 at m2048 along with pass10 and pass11, loading them with json.



v3 flips more under margin shrinkage (59 vs 48 at k=2) despite similar rms, so v1 looks better; I'm next comparing it against the older m2048 and m1536 runs on the same stress scale.



## tool_use: run_python

```json
{
 "code": "import json\nbl=np.load('base_logits.npy'); ids=json.load(open('all_ids.json'))\ndef stress(lg, ks=(1,1.5,2,3)):\n    pb=bl.argmax(1); d=lg-bl\n    return [int(((bl/k+d).argmax(1)!=pb).sum()) for k in ks]\nfor p in ['pass_10','pass_11','pass_12','pass_3','pass_27','pass_28']:\n    try:\n        pr=json.load(open(p+'/predictions.json')); idx={r['id']:i for i,r in enumerate(pr)}\n        if len(pr)<1000: print(p,len(pr)); continue\n        key='logits_edited' if 'logits_edited' in pr[0] else None\n        lg=np.array([pr[idx[i]][key] for i in ids])\n        print(p, stress(lg), np.sqrt(((lg-bl)**2).mean()).round(4))\n    except Exception as e: print(p,repr(e)[:100])\n"
}
```

