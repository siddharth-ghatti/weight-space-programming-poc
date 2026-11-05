## reasoning summary

I'm working out how sequence length affects the training data: sequences are mostly long, with a median around 195 tokens, while practice sequences are shorter, around 85. Early positions carry more error, and in short sequences they make up a larger share, so a 1/len² weighting would lean toward those early tokens, which means roughly 960 tokens in positions 1–10 across 96 sequences. I'm also considering whether adding short paragraphs to the data would better match the practice distribution.

I'm weighing whether to pull data from short-to-medium sequences, since the test set matches the practice distribution and the final noise is driven by short sequences. I'm considering a batch of 32 sequences in the 40–80 token range, though the gain looks modest because most tokens there sit at positions where error is already similar to long sequences. A uniform weighting per sequence length might be a more reasonable fix than targeting early positions alone.

I'm weighing whether to stop at m1536 and accept the roughly 17% reduction, or push to m1792 or m2048 for more margin. Re-weighting the training data looks like it might only buy about 5-10%, so the approximation is close to its limit. I'm also thinking about how the scoring function rewards fewer parameters among passing models, which bears on how much safety margin is worth giving up.

I'm now checking the flip behavior on the practice set: among low-margin items (below 0.5), how often the base model is right versus the candidate, which gives an in-sample read that I'd expect to be somewhat optimistic on the test set.



Every training sequence runs at least 120 tokens (median 195) against a practice median near 85, and early positions carry 20–40% more error, so there's no single fixable hotspot. Before spending passes on re-weighting, I'm checking how often flips between base's top-1 and top-2 classes are actually correct among low-margin practice items.



## tool_use: run_python

```json
{
 "code": "exec(open('common.py').read())\ns=np.argsort(-BL,1); top1=s[:,0]; top2=s[:,1]; srt=np.sort(BL,1); marg=srt[:,-1]-srt[:,-2]\nfor lo,hi in [(0,.25),(.25,.5),(.5,1),(1,2),(2,99)]:\n    sel=(marg>=lo)&(marg<hi)\n    print(f'{lo}-{hi}: n={sel.sum()} P(top1 right)={np.mean(y[sel]==top1[sel]):.2f} P(top2 right)={np.mean(y[sel]==top2[sel]):.2f}')\nlg=np.load('lg_pass26.npy'); f=lg.argmax(1)!=bp\nprint('flips', f.sum(), 'base right', (bp[f]==y[f]).sum(), 'edit right', (lg.argmax(1)[f]==y[f]).sum())\nprint('flip pairs', [(LABELS[a][:4],LABELS[b][:4]) for a,b in zip(bp[f], lg.argmax(1)[f])])\n"
}
```

