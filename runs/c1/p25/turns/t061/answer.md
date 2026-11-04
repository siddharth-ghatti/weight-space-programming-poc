## reasoning summary

Pruning down to 2560 scores worse than dropping a layer, at 97.36 against roughly 97.7 for the base. The drift is large, about 0.65 of the centered variance. I suspect the mean-folding is off, since the proxy activations may not match the real FFN inputs, which could be shifting the folded bias.

I'm weighing whether the drift could be the mean-folding bias, though a uniform shift wouldn't matter for z-scored features. Pruning the FFN isn't viable without accurate activations, so I'm settling on m=3072. Comparing k=9 (M8) with k=10 (M9) on the 1024 subset, they sit closer together than on the full set, and I want to quantify that more carefully with more seeds and the real GELU kernel.

Next I'm weighing whether a carry channel could hold nonlinear information, since token-level features averaged over tokens differ from a nonlinear function of the averages. I'm also considering whether multiple pool heads with learned, weighted attention would do better than a uniform mean.

I'm weighing whether a kernel over the M8 and CLS8 features could help, though an earlier quad version did worse. My current read is k=9 at about 97.65 versus k=10 at about 98.05, so I want to check the gap with a paired analysis across five seeds on the full set, using the GELU-kernel readouts. I'll also estimate how likely each candidate is to fall below the start model's 98.31 with a bootstrap.

I'm working out how many test items would make 98.31% come out cleanly, since the exact test size is unknown. Candidate sizes like 59, 118, or 177 all round to that figure, so I'll check a range of N up to a few thousand to see which fit.

I'm now checking which total test sizes would make 98.31% come out cleanly, restricting to multiples of 8 for balanced classes, and enumerating them in code.



Pruning to m=2560 costs more accuracy than dropping a layer (M9 at 97.36 versus 97.71–97.81 for base M9), so m stays 3072 and the decision is k=9 or k=10. I'm estimating test risk next via a 5-seed paired CV of the GELU-kernel candidates against the start readout.



## tool_use: run_python

```json
{
 "code": "Ns=[N for N in range(160,3001,8) if any(abs(round(100*k/N,2)-98.31)<1e-9 for k in range(int(.97*N),N+1))]\nprint(len(Ns), Ns[:40])\nNs_all=[N for N in range(100,3001) if any(abs(round(100*k/N,2)-98.31)<1e-9 for k in range(int(.97*N),N+1))]\nprint(len(Ns_all), Ns_all[:30])"
}
```

