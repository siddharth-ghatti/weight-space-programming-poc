## reasoning summary

The layers look fairly uniform, with the middle ones (roughly 3 through 8) showing the most redundancy. Before going further, I want to write down the key findings so far so they don't get lost.



Every layer shifts the token states by a similar amount (cosine around 0.92 in layers 2–8), so none stands out as skippable. I'm saving these findings to notes.md before the next experiment.



## tool_use: run_python

```json
{
 "code": "s=open('notes.md').read()\ns+='''\n## Run 2 findings (handoff 2)\n- E1 1-layer design is a dead end: E1 pooled feats add nothing over M0 (even leaky). Honest 1-layer ceiling ~97.7% (bag-of-embeddings kernel). Too risky.\n- RBF kernel exp(g(S-1)) on cosnorm, g=0.5-1, lam .003-.03 is the best kernel. M0 384-PCA dims == full M0.\n- Honest (KEEP dims) kernel 10f: M_l alone ~97.0-97.5 for l<=8; M9 98.4, M10 98.2, M11 98.1. M0+M8 98.3, M0+M9 98.5, M0+M10 98.66.\n  => need ~9-10 content layers of real BERT. Last kept layer only matters at CLS => it is always the pool layer (rewrite fully).\n- pass_5 (m=1024 std-pruned, layers0-10) kernel on M10: 94.2 vs base 97.6 (1024 ids). Pruning by std kills it.\n- token change per layer roughly uniform (cos in/out ~0.92 for layers 2-8, 0.86 L0, 0.88 L9).\n'''\nopen('notes.md','w').write(s)\n"
}
```

