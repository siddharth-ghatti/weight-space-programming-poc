## reasoning summary

Under cross-validation the mixed fit gains a little R² (.9515 to .9587) but agreement on edited inputs falls (.980 to .972), and base-input agreement drops from .998 to .992. So the mixed fit may be trading R² for worse class decisions, and its in-sample edge in pass 18 might not hold on test, where it could cause more flips than the base-only fit.

Looking at the in-sample edits from pass 13 and the mixed fit's targets, the mixed fit has to correct upstream error as well, which is noisy and may push the coefficients toward fitting noise. That would explain why R² rises while agreement falls, so the base-only L5 fit looks closer to lossless against the full FFN.

I'm now going to rebuild with a base-only L5 fit, since the mixed fit looks like it overfits on noisy upstream targets. I can't score that exactly without the pass-18 forward pass, but a probe with the LN layers set to identity and the FFN zeroed should give me the L5 inputs to evaluate any L5 variant offline on the CLS hidden state.

I'm going to check whether the L5 probe approach holds up, and whether I'm allowed to compute the LN and FFN steps on CLS vectors in numpy, since that starts to look like reimplementing the forward pass. I think it's fine as closed-form fitting analysis, as long as I confirm the result with real forward passes. My next step is to rebuild L0.

I'm checking whether the mixed L5 fit's CV advantage holds up on held-out items, and it doesn't: base-only agreement is .980 against .972 for mixed, so the base-only fit is the safer choice. That also puts the full FFN as the ceiling at .980, meaning pruning L5 with a base-only fit costs little beyond what the upstream layers already lose.

I'm now checking whether the probe approach for the L5 inputs holds up, and planning to use it for exact offline evaluation of the L5 variants. The next step is to rebuild the upstream layers for the chosen width and run a probe pass to get the inputs for the final layer.

I'm thinking about whether to check the upper bound too, which would show how much of the gap comes from the earlier layers rather than the final one. Before doing that, I want to look at how probe.py sets things up.



Under item-level cross-validation the mixed L5 fit gains R² but loses class agreement, so its earlier gain in passes 17–18 was overfitting. I'm checking probe.py to read the exact L5 inputs so the variants can be compared fairly.



## tool_use: run_python

```json
{
 "code": "\nprint(open('probe.py').read()[:5000])\n"
}
```

