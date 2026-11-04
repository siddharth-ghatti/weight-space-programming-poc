## reasoning summary

The fit looks good: the table reaches full training accuracy, which is expected given the cross-validation sits around 97%. Unseen random tokens mostly come out below 0.14, with a few higher, likely because they resemble something already known. Now I'm working out the scaling, where choosing a = 0.3 would bring the 99th percentile norm to about 2.

Now I'm checking whether the spread of the aggregated scores would hold up as a clean margin, while keeping the pooler scaling small enough that the logits stay manageable.

I'm weighing the scale of the aggregated pooled values, and whether a larger shared multiplier G would still keep the final classifier logits in a reasonable range.

I'm working out how the CLS token's context gets built: it averages the per-class values of every unmasked token, where each one is the scaled and centered output over its own mean and spread. The open question is whether the fitted outputs are actually centered across classes, since the mean depends on their sum, so I'm considering forcing exact centering by subtracting the column means.

Now I'm centering the fitted outputs by subtracting each neuron's mean, so every output vector sums to zero and the per-token LN mean shift stays negligible across classes. Since argmax is unchanged, the only effect is keeping the context values tidy.

I'm saving the arrays so the build script can be re-run later.



The fit is very accurate (about 0.12% weighted error, and 99% of unseen tokens come out with norm below 0.14). Next I'll measure the averaged class scores to choose the residual scale, aggregation gain and pooler scale.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom lib import *\nP,y,ids,X = load_data(practice, tokenize, LABELS)\nS=np.load('S_final.npy')\nn=np.array([len(t) for t in ids]); Xtf=sp.diags(1.0/n)@X\na=0.3\nm = a*(Xtf@S)\nprint('m abs pct', np.percentile(np.abs(m),[50,90,99,100]))\nsrt=np.sort(m,1); marg=srt[:,-1]-srt[:,-2]\nprint('top1-top2 margin pct', np.percentile(marg,[1,5,50]))\nprint('||m|| pct', np.percentile(np.linalg.norm(m,axis=1),[50,99,100]))"
}
```

