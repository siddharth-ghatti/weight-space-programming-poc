## reasoning summary

I'm checking whether the 0.99967 training accuracy is a real flip, since earlier the argmax came out at 1.0 for every item. My guess is tanh saturation causing ties in floating point, which would be odd because tanh should preserve argmax, so I'm testing that next.

I'm now picking which 512 practice items to verify on: I'll take the first 512 in practice order so I can match the predictions back to the rows in C. I still need to check ids_order.json to confirm that order matches base_feats.



The expected outputs match the labels on 99.97% of training points, with one sentence flipped by a small leak. I'm now running a predict pass on 512 practice ids to verify the saved model.



## tool_use: run_python

```json
{
 "code": "import json\nids=json.load(open('ids_order.json')); print(len(ids), ids[:3])\npr=practice(); print(pr[0]['id'], [p['id'] for p in pr[:3]]==ids[:3])\n# choose 512 spread across: every ~3rd\nsel=ids[1::3][:512]; print(len(sel))\njson.dump(sel,open('verify_ids_v4.json','w'))\nprint(json.dumps(sel))\n"
}
```

