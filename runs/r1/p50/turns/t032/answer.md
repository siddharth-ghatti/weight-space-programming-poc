## reasoning summary

I'm noticing that pruning layers 0 through 4 leaves only about a 5.3% relative error in the CLS output, yet layer 5, which is unpruned, pushes that error up to 23.7%. That's roughly a 4.5x amplification, so layer 5 seems to be especially sensitive, and I'm wondering whether its attention is the cause.

I'm wondering whether the CLS token's own error at layer 4 understates the problem, since layer 5's attention could be pulling in larger errors from the other tokens. It could also be that the CLS vector has large constant dimensions, so a relative error measured against its full norm looks small. I'll try measuring the error against the deviation from the mean CLS output to see where the noise is coming from.



Pruning layers 0-4 leaves the `<s>` vector at 5% error, rising to 24% after the untouched layer 5, likely from amplification or a large constant component hiding the error. I'm re-measuring against the varying part of each representation next.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nrel = lambda a,b: (np.linalg.norm(a-b,axis=1)/np.linalg.norm(b,axis=1)).mean()\nrelc = lambda a,b: np.linalg.norm(a-b)/np.linalg.norm(b-b.mean(0))\nfor l in range(7):\n    b = np.load(WORK/f'pass_2/cls_hidden_L{l}.npy')\n    print(l, 'cls norm', np.linalg.norm(b,axis=1).mean().round(2), 'varying-part norm', np.linalg.norm(b-b.mean(0),axis=1).mean().round(2), 'top dims |mean|', np.sort(np.abs(b.mean(0)))[-3:].round(1))\nc5 = np.load(WORK/'pass_11/cls_hidden_L5.npy'); b5 = np.load(WORK/'pass_2/cls_hidden_L5.npy')\nc6 = np.load(WORK/'pass_11/cls_hidden_L6.npy'); b6 = np.load(WORK/'pass_2/cls_hidden_L6.npy')\nprint('A: centered rel err L5', relc(c5,b5), 'L6', relc(c6,b6))\nc6m = np.load(WORK/'pass_10/cls_hidden_L6.npy'); print('m1536: centered rel L6', relc(c6m,b6))\nc5d = np.load(WORK/'pass_3/cls_hidden_L5.npy'); print('drop4: centered rel L6', relc(c5d,b6))"
}
```

