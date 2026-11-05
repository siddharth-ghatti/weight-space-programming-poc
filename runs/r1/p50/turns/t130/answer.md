## reasoning summary

Practice confidence averages about .877 against .925 accuracy, so the base model isn't overconfident there, and its logits look fairly modest. On test, with accuracy at .775, a calibrated model would average around .775 confidence, which implies a temperature near 1.5. Test is usually overconfident, so I'd expect something a bit lower, maybe 1.2 to 1.5. The shrinkage picture doesn't quite hold, though, since the test items are different rather than shrunken practice items.

Checking the margin change distribution next: it's heavy-tailed, with the 99th percentile around 1.2 logits, so some items shift a lot. I want to see whether those large changes cluster on short sequences, since that would affect how I weight the compensation fit.



The base model's confidence is close to its accuracy on practice, and implies a calibration factor near 1.5 if it's roughly calibrated on test. The margin changes have a heavy tail, so I'm checking whether the large errors cluster in short paragraphs.



## tool_use: run_python

```json
{
 "code": "import json\nbl=np.load('base_logits.npy'); ids=json.load(open('all_ids.json'))\nlens=json.load(open('lens.json')); print(type(lens), (list(lens.items())[:2] if isinstance(lens,dict) else lens[:3]))\nL=np.array([lens[i] for i in ids]) if isinstance(lens,dict) else np.array(lens)\nlg=np.load('lg_pass25.npy'); d=lg-bl\nrms_item=np.sqrt((d**2).mean(1))\npb=bl.argmax(1); s=np.sort(bl,1); marg=s[:,-1]-s[:,-2]\nbins=[0,40,60,80,100,150,200,300,600]\nfor a,b in zip(bins[:-1],bins[1:]):\n    m=(L>=a)&(L<b)\n    if m.sum(): print(f'len {a}-{b}: n={m.sum():4d} rms={np.sqrt((rms_item[m]**2).mean()):.3f}  flips={(lg[m].argmax(1)!=pb[m]).sum()}  marg<.5 frac={(marg[m]<.5).mean():.3f}')\nprint('corr log len vs log rms', np.corrcoef(np.log(L),np.log(rms_item))[0,1])\n"
}
```

