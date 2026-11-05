## reasoning summary

The edited model's flips in the low-margin bin look systematically harmful: in pass_10, nearly all the flips broke items that were correct before. Noise on margin is slightly negative there, so I'm wondering whether the drift pulls predictions toward one dominant class, possibly 'strategy'. I'll check the direction of the flips with a confusion breakdown.

I'm thinking the drift might push predictions toward one dominant class, like 'strategy', which would hurt the minority classes and macro-F1. If the drift is a systematic logit bias, I could fix it cheaply by adjusting the out_proj bias with per-class offsets, so I'm checking the mean change in logits per class.



In the lowest-margin bin, base is right only 62% of the time, yet 22 of the 23 m1536 flips broke correct items, which suggests systematic drift rather than noise. I'm checking the flip directions and the mean per-class logit shift next, since a bias shift could be fixed through out_proj.bias.



## tool_use: run_python

```json
{
 "code": "exec(open('common.py').read())\nfor d in ['pass_10','pass_12']:\n    lg = np.load(d+'/logits.npy'); p = lg.argmax(1); fl = p!=bp\n    dm = lg-BL\n    print(d, 'mean dm per class', dm.mean(0).round(3), 'centered', (dm.mean(0)-dm.mean()).round(3))\n    from collections import Counter\n    print('  flips base->ed', Counter(zip(bp[fl].tolist(), p[fl].tolist())))\n    print('  flips truth', Counter(zip(y[fl].tolist(), bp[fl].tolist(), p[fl].tolist())))\n    # pred class counts\n    print('  counts base', np.bincount(bp,minlength=4), 'ed', np.bincount(p,minlength=4), 'y', np.bincount(y,minlength=4))"
}
```

